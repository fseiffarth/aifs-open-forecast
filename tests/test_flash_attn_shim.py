"""The flash_attn shim must match plain dense attention with the same mask."""

import math

import pytest
import torch

from flash_attn import flash_attn_interface as fai
from flash_attn import flash_attn_func


def reference(q, k, v, scale=None, causal=False, window=(-1, -1), softcap=0.0, alibi=None):
    q, k, v = (t.transpose(1, 2).double() for t in (q, k, v))  # (B, H, S, D)
    sq, sk, d = q.shape[2], k.shape[2], q.shape[3]
    scores = q @ k.transpose(-1, -2) * (scale or 1 / math.sqrt(d))
    if softcap:
        scores = torch.tanh(scores / softcap) * softcap
    qi = torch.arange(sq).unsqueeze(1) + (sk - sq)
    kj = torch.arange(sk).unsqueeze(0)
    if alibi is not None:
        scores = scores - alibi.double().view(1, -1, 1, 1) * (qi - kj).abs()
    allowed = torch.ones(sq, sk, dtype=torch.bool)
    if window[0] >= 0:
        allowed &= kj >= qi - window[0]
    if window[1] >= 0:
        allowed &= kj <= qi + window[1]
    if causal:
        allowed &= kj <= qi
    out = torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1) @ v
    return out.transpose(1, 2)


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    # several blocks per sequence so block boundaries are exercised
    monkeypatch.setattr(fai, "_QUERY_CHUNK", 7)


def qkv(sq=40, sk=None, b=2, h=3, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    sk = sk or sq
    return (torch.randn(b, sq, h, d, generator=g), torch.randn(b, sk, h, d, generator=g),
            torch.randn(b, sk, h, d, generator=g))


@pytest.mark.parametrize("window", [(-1, -1), (5, 5), (0, 9), (12, -1)])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_dense_reference(window, causal):
    q, k, v = qkv()
    out = flash_attn_func(q, k, v, causal=causal, window_size=window)
    torch.testing.assert_close(out.double(), reference(q, k, v, causal=causal, window=window), atol=1e-5, rtol=1e-5)


def test_cross_attention_lengths():
    q, k, v = qkv(sq=17, sk=40)
    out = flash_attn_func(q, k, v, window_size=(6, 6))
    assert out.shape == q.shape
    torch.testing.assert_close(out.double(), reference(q, k, v, window=(6, 6)), atol=1e-5, rtol=1e-5)


def test_softcap_alibi_and_scale():
    q, k, v = qkv()
    slopes = torch.tensor([0.1, 0.2, 0.4])
    out = flash_attn_func(q, k, v, softmax_scale=0.3, softcap=5.0, alibi_slopes=slopes, window_size=(8, 8))
    ref = reference(q, k, v, scale=0.3, softcap=5.0, alibi=slopes, window=(8, 8))
    torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-5)


def test_keeps_input_dtype():
    q, k, v = (t.to(torch.bfloat16) for t in qkv())
    assert flash_attn_func(q, k, v).dtype == torch.bfloat16


def test_rejects_unsupported_options():
    q, k, v = qkv()
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, dropout_p=0.1)
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, return_attn_probs=True)
