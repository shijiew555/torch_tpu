import faulthandler
faulthandler.dump_traceback_later(150, exit=True)
import torch
import torch_tpu  # noqa: F401
from torch._functorch import config as fc
print("view_replay_for_aliased_outputs", getattr(fc, "view_replay_for_aliased_outputs", None))

def f(x):
    y = x * 2
    return y, y.view(2, 8)

x = torch.arange(16.0).view(4, 4).to("tpu")
a, b = torch.compile(f, backend="tpu")(x)
print("a base", a._base is not None, "b base", b._base is not None, "b._base is a", b._base is a)
print("ptrs", a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
