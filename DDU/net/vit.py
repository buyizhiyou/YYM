#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vit.py
@Time    :   2023/11/15 14:32:44
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''
"""vit for cifar10 32x32"""

import torch
from torch import nn
import torchsummary
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from net.spectral_normalization.spectral_norm_official import spectral_norm
from net.extra import ProjectionHead


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),  #ADD
            nn.Dropout(dropout))  #MLP

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, wrapped_fc, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads  #512
        project_out = not (heads == 1 and dim_head == dim)  #True

        self.heads = heads
        self.scale = dim_head**-0.5  #0.125

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = wrapped_fc(nn.Linear(dim, inner_dim * 3, bias=False))
        self.to_out = nn.Sequential(wrapped_fc(nn.Linear(inner_dim, dim)), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  #tuple ,len=3,torch.Size([4, 65, 512])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  #torch.Size([4, 8, 65, 64])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):  #Encoder

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, wrapped_fc, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  #6
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, wrapped_fc, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self,
                 spectral_normalization=True,
                 mod=True,
                 temp=1.0,
                 image_size=32,
                 patch_size=4,
                 num_classes=10,
                 dim=512,
                 depth=6,
                 heads=8,
                 mlp_dim=512,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)  #4x4

        wrapped_fc = spectral_norm if spectral_normalization else nn.Identity()

        self.temp = temp

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  #64
        patch_dim = channels * patch_height * patch_width  #48
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            wrapped_fc(nn.Linear(patch_dim, dim)),
        )  #划分patch
        #torch.einsum("bhif, bhjf->bhij", q, k)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  #torch.Size([1, 65, 512]) nn.Parameter类，加入参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  #torch.Size([1, 1, 512])
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, wrapped_fc, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), wrapped_fc(nn.Linear(dim, num_classes)))
        self.projection_head = ProjectionHead(512, 256)

        self.embedding=None
        self.feature=None
       


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  #[1,1,512]->[b,1,512]
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  #使用第0个cls_token作为最终Mlp head 的输入

        x = self.to_latent(x)

        self.embedding=self.projection_head(x)
        self.feature=x

        return self.mlp_head(x) / self.temp


def vit(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ViT(spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model


if __name__ == "__main__":
    b, c, h, w = 4, 3, 32, 32
    device = "cuda"
    x = torch.randn(b, c, h, w).to(device)
    net = ViT().to(device)
    out = net(x)
    # out.mean().backward()
    torchsummary.summary(net, (c, h, w), device=device)
    # print(out.shape)
