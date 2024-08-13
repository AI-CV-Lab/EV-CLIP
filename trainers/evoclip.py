## EVoCLIP


import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange, repeat
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy, vn_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .prompt_generator import EVoPrompt
from .prompt_generator import SimpleTokenizer as _Tokenizer

import clip

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, device):
    backbone_name = cfg.MODEL.BACKBONE.IMAGE.NAME
    model, _ = clip.load(backbone_name, device=device)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts=[]):
        if len(tokenized_prompts):
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            x = self.token_embedding(prompts).type(self.dtype)
            
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), prompts.argmax(dim=-1)] @ self.text_projection
        return x


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.MODEL.COOP.N_CTX
        ctx_init = cfg.MODEL.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.MODEL.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.MODEL.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        
        return prompts
    
    
class TemporalAttention(nn.Module):
    def __init__(self, clip_model, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.dtype = clip_model.dtype

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # only calculate attention of CLS token
        dots = torch.matmul(q[:, :, :1, :], k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out).squeeze(dim=1)

class MovingAverage(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()

        self.frames = cfg.INPUT.FRAMES
        self.device = device
        self.alpha = nn.Parameter(torch.arccos(torch.ones([])*(2/self.frames-1)), requires_grad=True).to(self.device) # initiate as 1/t

    def forward(self, x):
        x = x.permute(0, 2, 1)    # [B, 512, T]
        b, _, _ = x.shape

        # Feature Normalization
        x = x / torch.norm(x, dim=1, keepdim=True).expand((b, 512, self.frames))
        
        exps = torch.tensor([((1-torch.cos((self.alpha+1)/2))**(self.frames-i-1))*torch.cos((self.alpha+1)/2) for i in range(self.frames)]).to(self.device)
        exps = exps / torch.sum(exps)

        return x @ exps


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.evo_enable = cfg.MODEL.EVO.ENABLE
        if self.evo_enable:
            self.image_prompt_learner = EVoPrompt(cfg)
        self.prompt_dec_type = cfg.MODEL.EVO.DEC_TYPE # Decoder type of the Prompt
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        
        self.coop_enable = cfg.MODEL.COOP.ENABLE
        if self.coop_enable:
            self.text_prompt_learner = TextPromptLearner(cfg, classnames, clip_model, device)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts.to(device)
        else:
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [cfg.DATASET.PROMPT.replace('_', c) for c in classnames]
            print(f"Prompts: {prompts}")
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            self.prompts = prompts
        
        self.text_encoder = TextEncoder(clip_model)
        self.temporal_aggregation_method = cfg.MODEL.TEMPORAL

        self.temporal_including_dec_types = ["Context", "Both"]
        self.temporal_token_dec_types = ["Context", "Both"]

    def forward(self, video):

        if self.evo_enable:
            if self.prompt_dec_type in self.temporal_including_dec_types:
                video, temporal = self.image_prompt_learner(video)    # Spatially Masked Video, Temporal Mask (B, T, 512)
            else:
                video = self.image_prompt_learner(video)
        
        # extract video_features
        B, C, T, H, W = video.size()
        image = video.permute(0, 2, 1, 3, 4).flatten(0, 1)                          # [B, C, T, H, W] -> [B*T, C, H, W]
        image_features = self.image_encoder(image.type(self.dtype))
        frame_tokens = rearrange(image_features, '(b t) e -> b t e', b=B,t=T)       # [B, T, 512]

        # If temporal token exists (Feature-Level)
        if self.prompt_dec_type in self.temporal_token_dec_types:
            frame_tokens = torch.concat([temporal, frame_tokens], dim=1)
        
        video_features = frame_tokens.mean(dim=1)                               # [B, 512]
        
        if self.coop_enable:
            prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts.to(self.device)
            text_features = self.text_encoder(prompts, tokenized_prompts)
        else:
            text_features = self.text_encoder(self.prompts.to(self.device))

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * video_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class EVoCLIP(TrainerX):
    """
        EVo-CLIP
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.EVO.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.IMAGE.NAME})")
        clip_model = load_clip_to_cpu(cfg, self.device)
        
        if cfg.TRAINER.EVO.PREC == "fp32" or cfg.TRAINER.EVO.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        # parameter freeze
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "temporal_aggregation" not in name and "video_cls_prompt" not in name:
                param.requires_grad_(False)
            if "omnivore" in name:
                param.requires_grad_(False)
                
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.EVO.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
        # nan detector
        torch.autograd.set_detect_anomaly(True)

    def forward_backward(self, batch):
        
        image, label = self.parse_batch_train(batch)
        
        if self.cfg.INPUT.CLIP_PREPROCESS:
            image = self._batch_clip_preprocessing(image)
        
        prec = self.cfg.TRAINER.EVO.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            loss.requires_grad_(True)
            self.model_backward_and_update(loss)
        
        if self.cfg.TEST.EVALUATOR in ["Classification", "Classification_TopK"]:
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }
        else:
            verb_acc, noun_acc = vn_accuracy(output, label)
            
            loss_summary = {
                "loss": loss.item(),
                "verb_acc": verb_acc.item(),
                "noun_acc": noun_acc.item(),
                "action_acc": compute_accuracy(output, label)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def _batch_clip_preprocessing(self, batch):
        """
            CLIP preprocessing
                1. Permute the dimension of the tensor. [B x C x T x H x W] -> [B x T x C x H x W]
                2. Apply the preprocessing for all the frames.
                    (It is necessary to change each frame to PIL image type to apply the function)
                3. Permute the dimension of the tensor. [B x T x C x H x W] -> [B x C x T x H x W]
        """
        batch = batch.permute(0, 2, 1, 3, 4)
        for i, video in enumerate(batch):
            for j, frame in enumerate(video):
                batch[i, j] = self.preprocess(T.functional.to_pil_image(frame))
        
        return batch.permute(0, 2, 1, 3, 4)