# Runs the new test bodies against the installed (unfixed) torch_tpu 0.3.0.
import torch, torch.nn.functional as F
torch.manual_seed(0); cols = torch.randn(2, 18, 15)
f = lambda d: F.fold(cols.to(d), output_size=(5, 6), kernel_size=(3, 2)).cpu()
print("fold max err", (f("cpu") - f("tpu")).abs().max().item())
torch.manual_seed(0); x = torch.randn(2, 3, 6, 7); g = torch.randn(2, 27, 20)
def u(d):
  xd = x.to(d).detach().requires_grad_(); F.unfold(xd, kernel_size=3).backward(g.to(d)); return xd.grad.cpu()
print("im2col bwd max err", (u("cpu") - u("tpu")).abs().max().item())
torch.manual_seed(0); x2 = torch.randn(5, 4, 3); g2 = torch.randn(4, 4, 3, 2)
def t(d):
  xd = x2.to(d).detach().requires_grad_(); xd.unfold(0, 2, 1).backward(g2.to(d)); return xd.grad.cpu()
print("unfold bwd max err", (t("cpu") - t("tpu")).abs().max().item())
img = torch.arange(144, dtype=torch.float32).reshape(1, 12, 12); fo = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
print("existing fold test max err", (fo(img) - fo(img.to("tpu")).cpu()).abs().max().item())
