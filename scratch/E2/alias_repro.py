"""Outputs of a compiled TPU graph that alias each other lose the alias.

Eager, aot_eager, and inductor keep `b` a view of `a`. backend="tpu"
returns two separate buffers when no input requires grad.
"""
import torch
import torch_tpu  # noqa: F401

print("torch_tpu:", torch_tpu.__file__)


def f(x):
    y = x * 2
    return y, y.view(2, 8), y[1:]


def check(device, backend, requires_grad=False):
    torch._dynamo.reset()
    x = torch.arange(16.0).view(4, 4).to(device).requires_grad_(requires_grad)
    fn = torch.compile(f, backend=backend) if backend else f
    a, b, c = fn(x)
    shared = (
        a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and a.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()
    )
    with torch.no_grad():
        a.add_(100)
    print(
        f"{device} backend={backend!s:9s} requires_grad={requires_grad!s:5s}"
        f" shared={shared!s:5s} b[0,:3]={b[0, :3].tolist()}"
        f" c[0,:3]={c[0, :3].tolist()}"
    )


check("cpu", None)
check("cpu", "aot_eager")
check("tpu", None)
check("tpu", "tpu")
check("tpu", "tpu", requires_grad=True)
