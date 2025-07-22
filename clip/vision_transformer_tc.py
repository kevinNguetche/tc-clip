"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from einops import rearrange
import torch
from torch import nn

from clip.model_utils import LayerNorm
from clip.transformer import Transformer

from pitome.utils import parse_keep_ratio  # switched from tome.utils


class TCVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, num_frames: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.pos_emb_type = design_details["positional_embedding_type"]
        if self.pos_emb_type == "space":
            self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        elif self.pos_emb_type == "joint":
            self.positional_embedding = nn.Parameter(scale * torch.randn(num_frames,
                                                                         (input_resolution // patch_size) ** 2 + 1, width))
        else:
            raise NotImplementedError

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # PiToMe scheduler ---------------------------------------------------
        self.num_layers = layers
        self.pitome_ratio = design_details["tome_ratio"]   # # float | tuple | list
        self._tome_info = {
            "ratio": None,            # list[float] populated per forward
            "size": None,
            "source": None,
            "trace_source": False,
            "class_token": False,
        }

    # -------------- helper (unchanged) ----------------
    def add_positional_embedding(self, x, B, T):
        if self.pos_emb_type == "space":
            x = x + self.positional_embedding.to(x.dtype)
        elif self.pos_emb_type == "joint":
            BT, N, width = x.size()
            x = x.reshape(B, T, N, width)
            x = x + self.positional_embedding.to(x.dtype)
            x = x.reshape(BT, N, width)
        else:
            raise NotImplementedError
        return x

    # ---------------- forward -------------------------
    def forward(self, x: torch.Tensor, return_layer_num=None, return_attention=False, return_source=False):
        # (patchify + CLS + positional embedding) – unchanged
        B, T, C, H, W = x.shape
        x = self.conv1(x.reshape(-1, C, H, W))
        x = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype).unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)
        x = self.add_positional_embedding(x, B, T)
        x = rearrange(x, '(B T) N D -> B (T N) D', B=B, T=T)
        x = self.ln_pre(x)

        # build keep‑ratio list per layer
        self._tome_info["ratio"] = parse_keep_ratio(self.num_layers, self.pitome_ratio)
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        self._tome_info["trace_source"] = return_source

        x, attns, source = self.transformer.forward_tc(x,
                                                       tome_info=self._tome_info,
                                                       layer_num_list=return_layer_num,
                                                       return_attention=return_attention,
                                                       return_source=return_source)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        cls_tokens = x[:, :, :T, :]
        context_tokens = x[:, :, T:, :]
        return cls_tokens, context_tokens, attns, source