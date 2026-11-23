import warnings
warnings.simplefilter("error", FutureWarning)
import torch
import torch_tpu  # noqa: F401
print("before:", torch.tpu._initialized)
x = torch.ones(2, device="tpu")
print("after tensor:", torch.tpu._initialized)
print("is_initialized:", torch.tpu.is_initialized() if False else "skip")
