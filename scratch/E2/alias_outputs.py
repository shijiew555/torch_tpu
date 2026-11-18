"""Compiled outputs that alias each other must stay aliased (eager semantics)."""
import sys
import faulthandler
faulthandler.dump_traceback_later(150, exit=True)

import torch
import torch_tpu  # noqa: F401

print("torch_tpu:", torch_tpu.__file__)


def f(x):
    y = x * 2
    return y, y.view(2, 8)


def check(device, backend):
    torch._dynamo.reset()
    x = torch.arange(16.0).view(4, 4).to(device)
    fn = torch.compile(f, backend=backend) if backend else f
    a, b = fn(x)
    same_storage = a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    a.add_(100)
    print(
        f"{device:4s} backend={backend!s:10s} share_storage={same_storage} "
        f"b[0,:3] after a.add_(100) = {b[0, :3].cpu().tolist()}"
    )


check("cpu", None)
check("cpu", "aot_eager")
check("cpu", "inductor") if "inductor" in sys.argv else None
check("tpu", None)
check("tpu", "tpu")
