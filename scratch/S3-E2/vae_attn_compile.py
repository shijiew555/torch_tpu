"""Compiled backward of the diffusers VAE attention block on TPU.

Usage: python vae_attn_compile.py <tpu|cpu> <tpu|aot_eager> [attn|mini]
"""
import sys
import traceback

import torch
import torch_tpu  # noqa: F401

dev, backend = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "attn"
torch.manual_seed(0)

if kind == "attn":
  from diffusers.models.attention_processor import Attention
  m = Attention(64, heads=1, dim_head=64, rescale_output_factor=1.0, eps=1e-6,
                norm_num_groups=8, residual_connection=True, bias=True,
                upcast_softmax=True, _from_deprecated_attn_block=True)
else:

  class Mini(torch.nn.Module):
    """view + transpose before a Linear, as in the VAE attention block."""

    def __init__(self):
      super().__init__()
      self.q = torch.nn.Linear(64, 64)

    def forward(self, x):
      b, c, h, w = x.shape
      y = x.view(b, c, h * w).transpose(1, 2)
      y = self.q(y)
      return y.transpose(-1, -2).reshape(b, c, h, w)

  m = Mini()

m = m.to(dev)
x = torch.randn(2, 64, 16, 16, device=dev, requires_grad=True)
f = torch.compile(m, backend=backend)
try:
  out = f(x)
  out.pow(2).mean().backward()
  print("OK", dev, backend, kind, "grad sum", x.grad.sum().item())
except Exception as e:  # noqa: BLE001
  print("FAIL", dev, backend, kind, type(e).__name__, str(e).splitlines()[0][:300])
  tb = traceback.format_exc().splitlines()
  print("\n".join(l for l in tb if "File " in l and ("torch_tpu" in l or "_decomp" in l or "_refs" in l))[:3000])
