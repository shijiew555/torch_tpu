"""torch.compile on CPU tensors only. Does it open the TPU device?"""
import os
import torch

def vfio_open():
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if "vfio" in os.readlink(f"/proc/self/fd/{fd}"):
                n += 1
        except OSError:
            pass
    return n

print("after import torch, vfio fds:", vfio_open())
x = torch.randn(3)
print("after eager CPU op, vfio fds:", vfio_open())
torch.compile(lambda t: t * 2, backend="eager")(x)
print("after torch.compile(backend='eager') on CPU, vfio fds:", vfio_open())
