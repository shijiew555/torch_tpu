"""channels_last output is not kept by some TPU kernels; a later view fails.

Usage: python cl_view.py <cpu|tpu> <eager|compile>
"""
import sys

import torch
import torch_tpu  # noqa: F401

dev, mode = sys.argv[1], sys.argv[2]


def f(x):
  # NHWC flatten: a valid view when x keeps channels_last strides.
  return torch.relu(x).permute(0, 2, 3, 1).view(2, -1)


x = torch.randn(2, 8, 4, 4).contiguous(memory_format=torch.channels_last).to(dev)
print("input strides", x.stride(), "relu strides", torch.relu(x).stride())
fn = torch.compile(f, backend="tpu" if dev == "tpu" else "aot_eager") if mode == "compile" else f
try:
  print("OK", dev, mode, tuple(fn(x).shape))
except Exception as e:  # noqa: BLE001
  print("FAIL", dev, mode, type(e).__name__, str(e).splitlines()[0][:250])
