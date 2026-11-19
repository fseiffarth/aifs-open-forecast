"""Blockwise sliding-window attention compatible with ``flash_attn.flash_attn_func``."""

import math
import os

import torch

_BLOCK = int(os.environ.get("AIFS_ATTN_BLOCK", "2048"))


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """q, k, v: (batch, seqlen, heads, headdim). Returns (batch, seqlen_q, heads, headdim).

    Semantics follow flash-attn: query i attends key j iff
    i + (Sk - Sq) - left <= j <= i + (Sk - Sq) + right   (-1 means unbounded).
    """
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported by the flash_attn shim")
    if dropout_p:
        raise NotImplementedError("dropout is not supported by the flash_attn shim (inference only)")

    out_dtype = q.dtype
    q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))  # (B, H, S, D)
    _, _, sq, d = q.shape
    sk = k.shape[2]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    left, right = window_size if window_size is not None else (-1, -1)
    offset = sk - sq
    if alibi_slopes is not None:
        slopes = alibi_slopes.float().to(q.device)
        slopes = slopes.view(1, -1, 1, 1) if slopes.dim() == 1 else slopes.view(slopes.shape[0], -1, 1, 1)

    out = torch.empty_like(q)
    for s in range(0, sq, _BLOCK):
        e = min(s + _BLOCK, sq)
        k0 = 0 if left < 0 else max(0, s + offset - left)
        k1 = sk if right < 0 else min(sk, e - 1 + offset + right + 1)
        if causal:
            k1 = min(k1, e - 1 + offset + 1)
        scores = torch.matmul(q[:, :, s:e], k[:, :, k0:k1].transpose(-1, -2)) * scale
        if softcap:
            scores = torch.tanh(scores / softcap) * softcap
        qi = torch.arange(s, e, device=q.device).unsqueeze(1) + offset
        kj = torch.arange(k0, k1, device=q.device).unsqueeze(0)
        if alibi_slopes is not None:
            scores = scores - slopes * (qi - kj).abs()
        allowed = torch.ones_like(qi - kj, dtype=torch.bool)
        if left >= 0:
            allowed &= kj >= qi - left
        if right >= 0:
            allowed &= kj <= qi + right
        if causal:
            allowed &= kj <= qi
        scores = scores.masked_fill(~allowed, float("-inf"))
        out[:, :, s:e] = torch.matmul(torch.softmax(scores, dim=-1), v[:, :, k0:k1])
        del scores
    return out.transpose(1, 2).to(out_dtype)
