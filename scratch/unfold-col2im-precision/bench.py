"""Times f32/bf16 fold, unfold fwd+bwd and Tensor.unfold bwd on TPU."""
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401

LABEL = sys.argv[1]
DEV = "tpu"
INNER = 10
REPS = 5


def sync():
  torch.accelerator.synchronize()


def make_cases(dtype):
  g = torch.Generator().manual_seed(0)
  r = lambda *s: torch.randn(*s, generator=g).to(DEV, dtype)
  cases = {}
  # ViT patchify, 224x224, patch 16.
  cols = r(8, 3 * 16 * 16, 196)
  cases["fold vit224 p16"] = lambda: F.fold(
      cols, output_size=224, kernel_size=16, stride=16)
  x1 = r(8, 3, 224, 224)
  g1 = r(8, 768, 196)
  def unfold_vit():
    x = x1.detach().requires_grad_()
    F.unfold(x, kernel_size=16, stride=16).backward(g1)
    return x.grad
  cases["unfold fwd+bwd vit224 p16"] = unfold_vit
  # 3x3 im2col, 64 channels, 56x56, padding 1.
  cols2 = r(8, 64 * 9, 56 * 56)
  cases["fold 64ch 56x56 k3"] = lambda: F.fold(
      cols2, output_size=56, kernel_size=3, padding=1)
  x2 = r(8, 64, 56, 56)
  g2 = r(8, 576, 3136)
  def unfold_im2col():
    x = x2.detach().requires_grad_()
    F.unfold(x, kernel_size=3, padding=1).backward(g2)
    return x.grad
  cases["unfold fwd+bwd 64ch 56x56 k3"] = unfold_im2col
  # Tensor.unfold: audio framing, 16 x 16000 samples, window 400, hop 160.
  x3 = r(16, 16000)
  g3 = r(16, 98, 400)
  def tensor_unfold():
    x = x3.detach().requires_grad_()
    x.unfold(1, 400, 160).backward(g3)
    return x.grad
  cases["Tensor.unfold bwd 16x16000 w400 h160"] = tensor_unfold
  return cases


for dtype in (torch.float32, torch.bfloat16):
  for name, fn in make_cases(dtype).items():
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    first = time.perf_counter() - t0
    for _ in range(5):
      fn()
    sync()
    per_call = []
    for _ in range(REPS):
      t0 = time.perf_counter()
      for _ in range(INNER):
        fn()
      sync()
      per_call.append((time.perf_counter() - t0) / INNER)
    print(json.dumps({"label": LABEL, "dtype": str(dtype).split(".")[1],
                      "case": name, "first_ms": first * 1e3,
                      "median_us": statistics.median(per_call) * 1e6}),
          flush=True)
