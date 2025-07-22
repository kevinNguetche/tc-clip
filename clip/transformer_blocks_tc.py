"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from collections import OrderedDict
from einops import rearrange
from typing import Optional
from timm.models.layers import trunc_normal_

import torch
from torch import nn

from clip.model_utils import LayerNorm, QuickGELU

from pitome.merge import pitome_vision, merge_source, merge_wavg   # ← PiToMe API
from pitome.utils import parse_keep_ratio                            # ← new helper

class TCAttentionBlock(nn.Module):
    """Temporal‑Contextual block with PiToMe summarisation."""

    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None,
                 i: int = 0, design_details: Optional[dict] = None):
        super().__init__()
        self.T = design_details['temporal_length']            # #frames per clip
        self.num_patches = 196                                # fixed for ViT‑B/16
        self.num_context_token = design_details['context_token_k']
        self.first_layer = (i == 0)

        self.attn = TCAttention(d_model, n_head,
                                first_layer=self.first_layer,
                                T=self.T,
                                seed_token_a=design_details["seed_token_a"],
                                local_global_bias=design_details['local_global_bias'])
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # --------------------------------------------------
    def forward(self, x, tome_info: dict, *, return_attention: bool = False):
        B, T, D = x.size(0), self.T, x.size(2)
        N = 1 + self.num_patches
        TN = T * N

        # (1) MHSA with seed selection
        x_attn, seed_index, attn = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x[:, :TN, :] + x_attn  # residual

        # (2) PiToMe summarisation on seed tokens
        context_tokens = self._summarize_context_tokens_pitome(x, seed_index, tome_info)

        # (3) FFN
        x = torch.cat([x, context_tokens], dim=1)
        x = x + self.mlp(self.ln_2(x))
        return x, attn

    # --------------------------------------------------
    def _summarize_context_tokens_pitome(self, x, seed_idx, tome_info):
        """Aggregate per‑frame seed tokens into k context tokens via PiToMe."""
        B, T, N, D = x.size(0), self.T, x.size(1) // self.T, x.size(-1)
        patch_tokens = rearrange(x, 'B (T N) D -> B T N D', T=T, N=N)
        patch_tokens = patch_tokens[:, :, 1:1 + self.num_patches, :]  # drop CLS
        Np = patch_tokens.size(2)
        Ns = seed_idx.size(2)

        # Flatten indices so each seed has a unique position in [0, T* Np)
        idx_offset = torch.arange(T, device=x.device).view(1, T, 1) * Np  # (1,T,1)
        flat_idx = (seed_idx + idx_offset).reshape(B, -1)                 # (B, T*Ns)

        patch_tokens = rearrange(patch_tokens, 'B T Np D -> B (T Np) D')
        seed_tokens = patch_tokens.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, T*Ns, D)

        # ---- PiToMe merge ----
        keep_ratio = tome_info["ratio"].pop(0)          # float in (0,1]
        keep_ratio = float(max(1e-2, min(0.95, keep_ratio)))
        merge_fn = pitome_vision(metric=seed_tokens, ratio=keep_ratio, class_token=False)

        if tome_info['trace_source']:
            tome_info['source'] = merge_source(merge_fn, seed_tokens, tome_info['source'])
        # we do a single PiToMe merge; size bookkeeping is unnecessary across layers
        seed_tokens, _ = merge_wavg(merge_fn, seed_tokens, None)
        # reset size so subsequent blocks start fresh
        tome_info['size'] = None
        context_tokens = seed_tokens  # now of shape (B, k, D) with k ≈ keep_ratio * T*Ns
        
        # map back to original patch index space
        if tome_info['trace_source']:
            source_full = torch.zeros(B, context_tokens.size(1), T * Np, device=x.device)
            source_full.scatter_(2, flat_idx.unsqueeze(1).expand(-1, context_tokens.size(1), -1), tome_info['source'])
            tome_info['source'] = source_full
        return context_tokens


class TCAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, T=16, seed_token_a=0.3, first_layer=False, local_global_bias=False):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.T = T
        self.num_patches = 196
        self.top_s = int(self.num_patches * seed_token_a)    # number of seed tokens in each frame
        self.first_layer = first_layer

        if not self.first_layer and local_global_bias:    # [num_heads, 2(local, global)]
            self.local_global_bias_table = nn.Parameter(torch.zeros(n_head, 1, 2))
            trunc_normal_(self.local_global_bias_table, std=.02)
        else:
            self.local_global_bias_table = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def get_seed_index(self, attn):
        B, T = attn.size(0) // self.T, self.T
        cls_attn = attn[:, :, 0, 1:1 + self.num_patches]  # [BT, head, num_patches]
        cls_attn = cls_attn.reshape(B, T, self.num_heads, self.num_patches)
        cls_attn = cls_attn.mean(dim=2)     # [B, T, num_patches]
        _, idx = torch.topk(cls_attn, self.top_s, dim=2, largest=True, sorted=True)  # [B, T, top_s]
        return idx

    def forward(self, x: torch.Tensor, return_attention=False):
        # x: [B, T*N, D] or [B, T*N+k, D]
        B, L, C = x.shape
        T, N = self.T, 1+self.num_patches
        BT, TN = B*T, T*N

        # in-projection
        q = self.q_proj(x[:, :TN, :])  # [B, T*N, D]
        k = self.k_proj(x)  # [B, T*N, D] or [B, T*N+k, D]
        v = self.v_proj(x)  # [B, T*N, D] or [B, T*N+k, D]

        # Repeat context tokens for temporal axis
        q = q.reshape(BT, N, C)
        if self.first_layer:
            k = k.reshape(BT, N, C)
            v = v.reshape(BT, N, C)
        else:
            k_local = k[:, :TN, :].reshape(BT, N, C)
            v_local = v[:, :TN, :].reshape(BT, N, C)
            k_context = k[:, TN:, :].unsqueeze(1).repeat(1, T, 1, 1).reshape(BT, -1, C)    # [B, k, D] -> [BT, k, D]
            v_context = v[:, TN:, :].unsqueeze(1).repeat(1, T, 1, 1).reshape(BT, -1, C)
            k = torch.cat([k_local, k_context], dim=1)   # [BT, N+k, D]
            v = torch.cat([v_local, v_context], dim=1)

        q = q.reshape(BT, q.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, num_heads, N, C // num_heads]
        k = k.reshape(BT, k.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, num_heads, N+k, C // num_heads]
        v = v.reshape(BT, v.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, num_heads, N+k, C // num_heads]

        # attention
        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)    # [BT, nhead, Nq, Nk]

        # Add Local-global bias
        if self.local_global_bias_table is not None:
            # expand [num_heads, 1, 2] -> [bt, nhead, Nq, Nk]
            local_bias = self.local_global_bias_table[:, :, 0:1].unsqueeze(0).repeat(BT, 1, attn.size(2), attn.size(2))
            global_bias = self.local_global_bias_table[:, :, 1:].unsqueeze(0).repeat(BT, 1, attn.size(2), attn.size(3) - attn.size(2))
            bias = torch.cat([local_bias, global_bias], dim=-1)
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(BT, q.size(2), C)

        # out-projection
        x = x.reshape(B, TN, C)
        x = self.out_proj(x)

        # Select top-s indices with CLS attention score
        index = self.get_seed_index(attn)

        if return_attention:
            return x, index, attn[:, :, :, :N]
        else:
            return x, index, None
