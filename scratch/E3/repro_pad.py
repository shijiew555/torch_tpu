import torch
import torch.nn.functional as F
x = torch.arange(16.).reshape(1, 1, 4, 4)
print("cpu:", F.pad(x, (-1, 2, 0, 0), mode="replicate"))
print("tpu:", F.pad(x.to("tpu"), (-1, 2, 0, 0), mode="replicate").cpu())
