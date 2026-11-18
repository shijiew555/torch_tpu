import torch
import torch.nn.functional as F
x = torch.randn(1, 3, 8, 8, device="tpu", requires_grad=True)
F.adaptive_max_pool2d(x, (2, 2)).sum().backward()
print("tpu grad sum:", x.grad.sum().item(), "(expected 12.0)")
