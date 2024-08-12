import torch
import torch.nn as nn
from einops import rearrange, repeat

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
# from .trainers.pretrained_models import *

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "UCF101_video": "a video of {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "HMDB51": "a photo of a {}.",
    "EpicKitchens100": "a photo of a {}.",
    "Kinetics400": "a photo of a {}"
}


def load_clip_to_cpu(cfg, device):
    backbone_name = cfg.MODEL.BACKBONE.IMAGE.NAME
    model, _ = clip.load(backbone_name, device=device)

    return model

@TRAINER_REGISTRY.register()
class ViFiCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
    
@TRAINER_REGISTRY.register()
class ImageZeroshotCLIP(ZeroshotCLIP):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.IMAGE.NAME})")
        clip_model = load_clip_to_cpu(cfg, self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image = image.squeeze(dim=2)
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class MeanPoolZeroshotCLIP(ZeroshotCLIP):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.IMAGE.NAME})")
        clip_model = load_clip_to_cpu(cfg, self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    # def model_inference(self, video):
    #     # extract video_features
    #     B, C, T, H, W = video.size()
    #     image = video.permute(0, 2, 1, 3, 4).flatten(0, 1)                          # [B, C, T, H, W] -> [B*T, C, H, W]
    #     image_features = self.clip_model.encode_image(image.type(self.clip_model.dtype))
    #     frame_tokens = rearrange(image_features, '(b t) e -> b t e', b=B,t=T)       # [B, T, 512]
    #     video_features = frame_tokens.mean(dim=1)                                   # [B, 512]
    #     video_features = video_features / video_features.norm(dim=-1, keepdim=True)
    #     logit_scale = self.clip_model.logit_scale.exp()
    #     logits = logit_scale * video_features @ self.text_features.t()
    #     return logits

    def i2v_pool(self, image_features):
        image_features = image_features.permute(0, 2, 1)
        b, _, t = image_features.shape

        # Feature Normalization
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True).expand((b, 512, t))

        expanded_feature = image_features.unsqueeze(-1).expand((-1, 512, t, t))
        low_tri = torch.tril(expanded_feature -  expanded_feature.permute(0, 1, 3, 2), diagonal=-1)

        weight = torch.tril(torch.ones((t, t), dtype=torch.float16), diagonal=-1).to(self.device)
        weight = weight / torch.sum(weight)

        action_feature = torch.sum((low_tri * weight.unsqueeze(0).expand(b, 512, t, t)).view(b, 512, -1), dim=2)
        action_feature = action_feature / torch.norm(action_feature, dim=1, keepdim=True).expand((b, 512))

        return action_feature

    def moving_avg(self, image_features):
        image_features = image_features.permute(0, 2, 1)    # [B, 512, T]
        b, _, t = image_features.shape

        # Feature Normalization
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True).expand((b, 512, t))

        alpha = 1/t
        exps = torch.tensor([((1-alpha)**(t-i-1))*alpha for i in range(t)], dtype=torch.float16).to(self.device)
        exps = exps / torch.sum(exps)

        return image_features @ exps
    
    def exponential_smoothing(self, image_features):
        image_features = image_features.permute(0, 2, 1)    # [B, 512, T]
        b, _, t = image_features.shape

        # Feature Normalization
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True).expand((b, 512, t))

        # alpha = 2/(t+1)
        alpha = 1/t
        coef1 = [1] + [alpha for i in range(t-1)]
        # coef1 = [alpha for i in range(t)]
        coef2 = [(1-alpha)**(t-i-1) for i in range(t)]

        exps = torch.tensor([c1*c2 for c1, c2 in zip(coef1, coef2)], dtype=torch.float16).to(self.device)
        exps = exps / torch.sum(exps)

        return image_features @ exps

    def model_inference(self, video):
        # extract video_features
        B, C, T, H, W = video.size()
        image = video.permute(0, 2, 1, 3, 4).flatten(0, 1)                          # [B, C, T, H, W] -> [B*T, C, H, W]
        image_features = self.clip_model.encode_image(image.type(self.clip_model.dtype))
        frame_tokens = rearrange(image_features, '(b t) e -> b t e', b=B,t=T)       # [B, T, 512]
        # video_features = frame_tokens.mean(dim=1) + self.i2v_pool(frame_tokens)     # [B, 512]
        # video_features = frame_tokens.mean(dim=1)                                     # [B, 512]
        video_features = self.moving_avg(frame_tokens)
        # video_features = self.exponential_smoothing(frame_tokens)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * video_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
