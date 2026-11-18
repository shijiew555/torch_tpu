import torch
torch.manual_seed(0)
x = torch.randn(5, 5, 5, dtype=torch.float64)
g = torch.randn(5, 5, 5, dtype=torch.float64)
def grad(x, g):
  x = x.detach().requires_grad_()
  torch.logcumsumexp(x, 0).backward(g)
  return x.grad
ref = grad(x, g)                                   # CPU float64 reference
c32 = grad(x.float(), g.float()).double()           # CPU float32
t32 = grad(x.float().to("tpu"), g.float().to("tpu")).cpu().double()
print("forward max err tpu f32:", (torch.logcumsumexp(x.float().to("tpu"), 0).cpu().double() - torch.logcumsumexp(x, 0)).abs().max().item())
print("grad max err cpu f32 vs f64:", (c32 - ref).abs().max().item())
print("grad max err tpu f32 vs f64:", (t32 - ref).abs().max().item(), "max |grad|:", ref.abs().max().item())
