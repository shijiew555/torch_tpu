"""Compiled forward+backward of a tiny diffusers AutoencoderKL on TPU.

Usage: python vae_compile.py <tpu|cpu> <tpu|aot_eager> [enc|dec|full]
"""
import sys
import traceback

import torch
import torch_tpu  # noqa: F401
from diffusers import AutoencoderKL

dev, backend = sys.argv[1], sys.argv[2]
part = sys.argv[3] if len(sys.argv) > 3 else "full"
torch.manual_seed(0)
m = AutoencoderKL(
    in_channels=3, out_channels=3, latent_channels=4,
    block_out_channels=(32, 64), norm_num_groups=8, layers_per_block=1,
    down_block_types=("DownEncoderBlock2D",) * 2,
    up_block_types=("UpDecoderBlock2D",) * 2).to(dev)
if part == "enc":
  mod, x = m.encoder, torch.randn(2, 3, 32, 32, device=dev)
elif part == "dec":
  mod, x = m.decoder, torch.randn(2, 4, 16, 16, device=dev)
elif part == "mid":
  mod, x = m.decoder.mid_block, torch.randn(2, 64, 16, 16, device=dev)
else:
  mod, x = (lambda a: m(a, sample_posterior=False).sample), torch.randn(2, 3, 32, 32, device=dev)
f = torch.compile(mod, backend=backend)
try:
  f(x).pow(2).mean().backward()
  print("OK", dev, backend, part)
except Exception as e:  # noqa: BLE001
  print("FAIL", dev, backend, part, type(e).__name__, str(e).splitlines()[0][:250])
  if "-v" in sys.argv:
    traceback.print_exc()
