import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
cols = torch.randn(2, 3 * 3 * 3, 36)  # F.fold input for a 3x3 kernel on 8x8 output
ref = F.fold(cols.double(), (8, 8), 3)
t = F.fold(cols.to("tpu"), (8, 8), 3).cpu()
print("F.fold (col2im) tpu f32 max err:", (t.double() - ref).abs().max().item())
print("F.fold cpu f32 max err:", (F.fold(cols, (8, 8), 3).double() - ref).abs().max().item())
