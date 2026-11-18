import torch
import torch.nn.functional as F
x = torch.randn(1, 2, 4, 4, 4)
ref = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
print("cpu shape:", tuple(ref.shape))
out = F.interpolate(x.to("tpu"), scale_factor=2, mode="trilinear", align_corners=False)
print("tpu max err:", (out.cpu() - ref).abs().max().item())
