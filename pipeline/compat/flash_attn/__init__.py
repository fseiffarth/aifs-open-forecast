"""Pure-PyTorch stand-in for the ``flash_attn`` package.

The AIFS checkpoint pickles a reference to ``flash_attn.flash_attn_interface.flash_attn_func``.
The real flash-attn needs an Ampere-or-newer GPU (sm_80+), so it cannot run on Turing cards
(e.g. RTX 2080) or on CPU. This shim implements the subset of ``flash_attn_func`` that AIFS
uses (non-causal, sliding-window, optional softcap/ALiBi) with blockwise attention so memory
stays bounded. It is only put on ``PYTHONPATH`` by the pipeline when the real package is unusable.
"""

__version__ = "2.8.3"  # anemoi-models checks for >= 2.6.0

from .flash_attn_interface import flash_attn_func  # noqa: E402,F401
