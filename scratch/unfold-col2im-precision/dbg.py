import torch, torch.nn.functional as F
torch.manual_seed(0); x2 = torch.randn(5, 4, 3); g2 = torch.randn(4, 4, 3, 2)
def t(d):
  xd = x2.to(d).requires_grad_(); xd.unfold(0, 2, 1).backward(g2.to(d)); return xd.grad.cpu()
a=t("cpu"); b=t("tpu"); print(a[:,0,0]); print(b[:,0,0])
x = torch.randn(1, 1, 4, 4); g = torch.randn(1, 4, 9)
def u(d):
  xd = x.to(d).requires_grad_(); F.unfold(xd, kernel_size=2).backward(g.to(d)); return xd.grad.cpu()
print(u("cpu")); print(u("tpu"))
print(torch.ops.aten.unfold_backward(g2, [5,4,3], 0, 2, 1)[:,0,0]); print(torch.ops.aten.unfold_backward(g2.to("tpu"), [5,4,3], 0, 2, 1).cpu()[:,0,0])
