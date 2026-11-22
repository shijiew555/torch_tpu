"""Output strides of group norm ops for a channels_last input: TPU vs CPU vs meta."""
import torch
import torch_tpu  # noqa: F401

aten = torch.ops.aten
N, C, H, W, G = 2, 64, 16, 16, 8
torch.manual_seed(0)
base = torch.randn(N, C, H, W).contiguous(memory_format=torch.channels_last)
gout = torch.randn(N, C, H, W)  # contiguous grad, as in the VAE graph
w = torch.randn(C)
b = torch.randn(C)


def run(dev):
  x = base.to(dev)
  if dev != "meta":
    assert x.stride() == base.stride(), x.stride()
  g = gout.to(dev)
  y, mean, rstd = aten.native_group_norm(x, w.to(dev), b.to(dev), N, C, H * W, G, 1e-6)
  gi, gw, gb = aten.native_group_norm_backward(g, x, mean, rstd, w.to(dev), N, C, H * W, G, [True, True, True])
  s = x * 2.0
  a = s + g
  print(f"{dev:5s} x={x.stride()} gn_fwd={y.stride()} gn_bwd_grad_input={gi.stride()} "
        f"mul={s.stride()} add(cl, contig)={a.stride()}")


for d in ("cpu", "meta", "tpu"):
  run(d)
