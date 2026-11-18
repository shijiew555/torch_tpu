"""Find which call opens the TPU when compiling a CPU-only function."""
import os
import traceback
import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal.device import _device_module as dm

cls = [v for v in vars(dm).values() if isinstance(v, type) and hasattr(v, "_lazy_init") and hasattr(v, "current_device")][0]
print("class:", cls.__name__)
for name in ("_lazy_init", "current_device", "device_count", "is_available", "get_rng_state", "set_rng_state", "manual_seed"):
    orig = getattr(cls, name, None)
    if orig is None:
        continue
    def make(name, orig):
        def wrapper(*a, **k):
            print(f"--- {name} called from:")
            print("".join(traceback.format_stack(limit=12)[:-1]))
            return orig(*a, **k)
        return staticmethod(wrapper) if isinstance(vars(cls).get(name), staticmethod) else classmethod(lambda c, *a, **k: wrapper(*a, **k))
    setattr(cls, name, make(name, orig))
x = torch.randn(3)
torch.compile(lambda t: t * 2, backend="eager")(x)
