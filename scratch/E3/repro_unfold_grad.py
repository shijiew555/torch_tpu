import torch
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
x = torch.randn(5, 5, 5, 5)
g = torch.randn(3, 5, 5, 5, 3)  # x.unfold(0, 3, 1) has shape (3, 5, 5, 5, 3)
def grad(x, g):
  x = x.detach().requires_grad_()
  x.unfold(0, 3, 1).backward(g)
  return x.grad
ref = grad(x.double(), g.double())
c = grad(x, g)
t = grad(x.to("tpu"), g.to("tpu")).cpu()
print("cpu f32 max err:", (c.double() - ref).abs().max().item())
print("tpu f32 max err:", (t.double() - ref).abs().max().item())
print("tpu f32 bf16-rounded-input err:", (grad(x, g.bfloat16().float()).double() - ref).abs().max().item())
