import torch
from torch.testing._internal.common_methods_invocations import op_db
op = [o for o in op_db if o.name == "gather"][0]
s = list(op.sample_inputs("cpu", torch.float32, requires_grad=True))[4]
print("input", tuple(s.input.shape), "args", [a.shape if isinstance(a, torch.Tensor) else a for a in s.args], s.args)
