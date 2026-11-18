"""Small TPU sanity check: add, matmul, and backward on TPU vs CPU."""
import torch
import torch_tpu

print("torch_tpu from", torch_tpu.__file__)
a = torch.randn(64, 64, requires_grad=True)
b = torch.randn(64, 64)
ta = a.detach().to("tpu").requires_grad_()
tb = b.to("tpu")
out = (ta @ tb + 1).relu().sum()
out.backward()
ref = (a @ b + 1).relu().sum()
ref.backward()
print("fwd close", torch.allclose(out.cpu(), ref, rtol=1e-2, atol=1e-1))
print("grad close", torch.allclose(ta.grad.cpu(), a.grad, rtol=1e-2, atol=1e-1))
print("max grad diff", (ta.grad.cpu()-a.grad).abs().max().item(), "mask flips", ((ta@tb+1).cpu().gt(0) != (a@b+1).gt(0)).sum().item())
