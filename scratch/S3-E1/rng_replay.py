import torch, torch_tpu  # noqa: F401
from torch import nn
from torch.utils.checkpoint import _get_device_states, _set_device_states

T = "tpu"
torch.manual_seed(3)
x = torch.ones(1000, device=T)
s = torch.accelerator.random.get_rng_state() if hasattr(torch.accelerator, "random") else None
print("accel get_rng_state:", None if s is None else (s.shape, s.dtype, s[:16]))
a = torch.rand(4, device=T)
torch.accelerator.random.set_rng_state(s)
b = torch.rand(4, device=T)
print("rand replay equal:", torch.equal(a, b))
torch.accelerator.random.set_rng_state(s)
d1 = nn.functional.dropout(x, 0.5)
torch.accelerator.random.set_rng_state(s)
d2 = nn.functional.dropout(x, 0.5)
print("dropout replay equal:", torch.equal(d1, d2))
# Path used by torch.utils.checkpoint
devs, states = _get_device_states(x)
print("checkpoint device states:", devs, [st.shape for st in states])
e1 = nn.functional.dropout(x, 0.5)
_set_device_states(devs, states, device_type=T)
e2 = nn.functional.dropout(x, 0.5)
print("checkpoint-path dropout replay equal:", torch.equal(e1, e2))
st0 = torch.accelerator.random.get_rng_state()
nn.functional.dropout(x, 0.5)
st1 = torch.accelerator.random.get_rng_state()
print("state changes after dropout:", not torch.equal(st0, st1))
st0 = torch.accelerator.random.get_rng_state()
torch.rand(4, device=T)
st1 = torch.accelerator.random.get_rng_state()
print("state changes after rand:", not torch.equal(st0, st1))
