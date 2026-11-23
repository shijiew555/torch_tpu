"""User-level repro: f32 F.fold / unfold backward error vs CPU on TPU."""
import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401

torch.manual_seed(0)
cols = torch.randn(2, 18, 15)
x4 = torch.randn(2, 3, 6, 7)
g4 = torch.randn(2, 27, 20)
x3 = torch.randn(5, 4, 3)
g3 = torch.randn(4, 4, 3, 2)


def fold(d):
  return F.fold(cols.to(d), output_size=(5, 6), kernel_size=(3, 2)).cpu()


def unfold_bwd(d):
  x = x4.to(d).detach().clone().requires_grad_()
  F.unfold(x, kernel_size=3).backward(g4.to(d))
  return x.grad.cpu()


def tensor_unfold_bwd(d):
  x = x3.to(d).detach().clone().requires_grad_()
  x.unfold(0, 2, 1).backward(g3.to(d))
  return x.grad.cpu()


for prec in ("default", "highest"):
  torch.set_float32_matmul_precision("high" if prec == "default" else prec)
  for name, fn in (("F.fold", fold), ("F.unfold backward", unfold_bwd),
                   ("Tensor.unfold backward", tensor_unfold_bwd)):
    err = (fn("cpu") - fn("tpu")).abs().max().item()
    print(f"matmul_precision={prec:8s} {name:24s} max abs err {err:.3g}")

torch.set_float32_matmul_precision("high")
compiled_fold = torch.compile(
    lambda c: F.fold(c, output_size=(5, 6), kernel_size=(3, 2)), backend="tpu")
err = (fold("cpu") - compiled_fold(cols.to("tpu")).cpu()).abs().max().item()
print(f"torch.compile(backend='tpu') F.fold max abs err {err:.3g}")
