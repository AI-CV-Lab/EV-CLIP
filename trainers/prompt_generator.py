import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import numpy as np


import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


def load_options(cfg):
    
    backbone_name = cfg.MODEL.EVO.ENC_NAME
    if backbone_name == "omnivore_swinT" or backbone_name == "omnivore_swinS":
        z_dim = 768
    elif backbone_name == "omnivore_swinB":
        z_dim = 1024
    else:
        z_dim = cfg.MODEL.EVO.ENC_OUT_DIM

    bias_flag = cfg.MODEL.EVO.BIAS
    frames = cfg.INPUT.FRAMES
    input_size = cfg.INPUT.SIZE

    if cfg.MODEL.EVO.ACT == "relu":
        act = nn.ReLU
    elif cfg.MODEL.EVO.ACT == "gelu":
        act = nn.GELU

    return z_dim, bias_flag, frames, input_size, act

"""
    Swin-Unet (Decoder)
    - Code Reference: https://github.com/HuCaoFighting/Swin-Unet
"""
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    [Partition the image into windows]
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    [Merge windows into an image]
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, expand_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand_scale = expand_scale
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, expand_scale*dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,C//(self.dim_scale**2))
        x= self.norm(x)

        return x

#------------------------------------------------------------------#

"""
    Swin-Unet based Mask Generator
"""

class EVo_Mask_Generator(nn.Module):       # HLDecS_Swin
    def __init__(self, cfg):
        super(EVo_Mask_Generator, self).__init__()

        z_dim, bias_flag, frames, input_size, act = load_options(cfg)
        self.z_dim = z_dim
        self.frames = frames
        self.input_size = input_size
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.intensity_revision = cfg.MODEL.EVO.SPATIAL_IR

        self.input_resolution = (7, 7)
        swin_input_resolution1 = tuple([r*2 for r in self.input_resolution])
        swin_input_resolution2 = tuple([r*8 for r in self.input_resolution])

        depth = 2
        num_heads = 8
        window_size = 7
        mlp_ratio = 4
        norm_layer=nn.LayerNorm

        # Linear Projection on Spatial Space
        self.projection2spatial = nn.Sequential(
                    nn.Linear(self.frames//2, self.frames//2, bias=bias_flag),
                    act(), 
                    nn.Linear(self.frames//2, 1, bias=bias_flag),
                    act(), 
                    nn.LayerNorm([self.input_resolution[0], self.input_resolution[1], self.z_dim, 1]),
                )

        # Upsampling for SWIN
        self.upsample1 = PatchExpand(self.input_resolution, dim=z_dim, dim_scale=2, expand_scale=2, norm_layer=norm_layer)
        self.upsample2 = PatchExpand(swin_input_resolution1, dim=z_dim//2, dim_scale=4, expand_scale=4, norm_layer=norm_layer)
        self.upsample3 = PatchExpand(swin_input_resolution2, dim=z_dim//8, dim_scale=4, expand_scale=4, norm_layer=norm_layer)

        # Swin Blocks
        self.block1 = nn.ModuleList([
            SwinTransformerBlock(dim=z_dim//2, input_resolution=swin_input_resolution1,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        self.block2 = nn.ModuleList([
            SwinTransformerBlock(dim=z_dim//8, input_resolution=swin_input_resolution2,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # Channel Projection
        if z_dim in [768, 1024]:
          self.channel_projection = nn.Sequential(
                      nn.Linear(z_dim//32, z_dim//96, bias=bias_flag),
                      act(), 
                      nn.Linear(z_dim//96, 1, bias=bias_flag),
                  )
        elif z_dim==128:
          self.channel_projection = nn.Sequential(
                      nn.Linear(z_dim//32, z_dim//32, bias=bias_flag),
                      act(), 
                      nn.Linear(z_dim//32, 1, bias=bias_flag),
                  )
        
        # Intensity Revision
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        # Projection to Spatial Space
        # [B, z_dim, T//2, 7, 7] -> [B, 7, 7, z_dim]
        z = self.projection2spatial(z.permute(0, 3, 4, 1, 2))

        # Patch Expanding (X2) + Swin Block
        # [B, 7, 7, z_dim] -> [B, 14, 14, z_dim//2]
        z = self.upsample1(z.squeeze(-1).reshape(-1,self.input_resolution[0]*self.input_resolution[1], self.z_dim))
        for blk in self.block1:
            z = blk(z.squeeze(-1))
        
        # Patch Expanding (X4) + Swin Block
        # [B, 7, 7, z_dim//2] -> [B, 56, 56, z_dim//8]
        z = self.upsample2(z)
        for blk in self.block2:
            z = blk(z.squeeze(-1))
        
        # Patch Expanding (X4) + Swin Block
        # [B, 56, 56, z_dim//8] -> [B, 224, 224, z_dim//32]
        z = self.upsample3(z)

        # Channel Projection
        # [B, 224, 224, z_dim//32] -> [B, 224, 224, 1]
        z = self.channel_projection(z)

        # SoftMax HighLighting
        mask = self.softmax(z.squeeze(-1).reshape(-1, self.input_size[0]*self.input_size[1])*self.logit_scale)
        max_probs = torch.max(mask, dim=-1).values.view(-1, 1, 1, 1, 1)
        min_probs = torch.min(mask, dim=-1).values.view(-1, 1, 1, 1, 1)
        mask = mask.reshape(-1, 1, 1, self.input_size[0], self.input_size[1])

        # Scaling key probs
        mask = torch.div(mask-min_probs, max_probs-min_probs+1e-8)

        # Expand the size
        mask = mask.expand(-1, 3, self.frames, -1, -1)

        return mask

class EVo_Context_Generator(nn.Module):     # HLDecT_AvgContext
    """
        [Temporal Mask for CLIP features]
    """
    def __init__(self, cfg):
        super(EVo_Context_Generator, self).__init__()
        
        z_dim, bias_flag, _, _, act = load_options(cfg)

        if "RN50" in cfg.MODEL.BACKBONE.IMAGE.NAME:
            latent_dim = 1024
        elif "L/14" in cfg.MODEL.BACKBONE.IMAGE.NAME:
            latent_dim = 768
        else:
            latent_dim = 512

        self.context_scale = nn.Parameter(torch.ones([])*0.001)
        
        self.feature_adaptation = nn.Sequential(
            nn.Linear(z_dim, latent_dim, bias=bias_flag),
            act(),
            nn.LayerNorm([latent_dim])
        )

    def forward(self, z):
        # [B, z_dim, T//2, 7, 7] -> [B, z_dim]
        z = torch.mean(z, [-3, -2, -1])

        # [B, z_dim] -> [B, z_dim]
        context_promt = self.feature_adaptation(z) * self.context_scale

        return context_promt.unsqueeze(1)

class EVo_Prompts(nn.Module):   # HLDecST_swin_avgcontext
    def __init__(self, cfg):
        super(EVo_Prompts, self).__init__()

        self.dec_s = EVo_Mask_Generator(cfg)
        self.dec_t = EVo_Context_Generator(cfg)

    def forward(self, z):

        mask_s = self.dec_s(z)     # [B, 3, N, 224, 244]
        context_promt = self.dec_t(z)     # [B, 3, N, 224, 244]

        return mask_s, context_promt

class Omnivore(nn.Module):
    """
        Pretrained Video Model
    """
    def __init__(self, cfg):
        super(Omnivore, self).__init__()
        
        self.model_name = cfg.MODEL.BACKBONE.VIDEO.NAME
        
        self.model = torch.hub.load("facebookresearch/omnivore", model=self.model_name)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.model.heads = nn.ModuleDict({"video":Identity()})
            
    def forward(self, x):
        features = self.model(x, input_type="video")
        return features

class EVoPrompt(nn.Module):
    def __init__(self, cfg):
        super(EVoPrompt, self).__init__()
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD

        self.enc_name = cfg.MODEL.EVO.ENC_NAME
        self.enc = torch.hub.load("facebookresearch/omnivore", model=self.enc_name)
        for name, param in self.enc.named_parameters():
            param.requires_grad = False
            
        dec_type = {
            'Mask': EVo_Mask_Generator,
            'Context': EVo_Context_Generator,
            'Both': EVo_Prompts,
        }
        self.decoder_type = cfg.MODEL.EVO.DEC_TYPE
        self.dec = dec_type[cfg.MODEL.EVO.DEC_TYPE](cfg)
        self.prompt_aggregation = cfg.MODEL.EVO.PROMPT_AGGREGATION
        
        if cfg.MODEL.EVO.PROMPT_INIT == "constant":
            for name, param in self.enc.named_parameters():
                nn.init.constant_(param, 0.02)
            for name, param in self.dec.named_parameters():
                nn.init.constant_(param, 0.02)
        elif cfg.MODEL.EVO.PROMPT_INIT == "zero":
            for name, param in self.enc.named_parameters():
                nn.init.zeros_(param)
            for name, param in self.dec.named_parameters():
                nn.init.zeros_(param)
        elif cfg.MODEL.EVO.PROMPT_INIT == "kaiming":
            if self.enc_name.startswith("omnivore"):
                for m in self.dec.modules():
                    if isinstance(m, nn.Conv3d):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.GroupNorm):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.ConvTranspose3d):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv3d):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.GroupNorm):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.ConvTranspose3d):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.enc_name.startswith("omnivore"):
            z = self.enc.trunk(x, out_feat_keys=['stage3'])[0]
        else:
            z = self.enc(x)
        
        if self.decoder_type in ["Both"]:
            """
                Mask & Context Prompts
            """
            p_s, p_t = self.dec(z)
            out = x*p_s
            return out, p_t

        elif self.decoder_type in ["Context"]:
            """
                Context Prompt
            """
            p_t = self.dec(z)
            return x, p_t

        else:
            """
                Mask Prompt
            """
            p_s = self.dec(z)
            return x*p_s
