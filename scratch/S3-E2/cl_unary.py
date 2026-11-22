"""Which elementwise ops keep channels_last strides on TPU (CPU keeps all)."""
import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401
x = torch.randn(2, 8, 4, 4).contiguous(memory_format=torch.channels_last)
ops = {"relu": torch.relu, "exp": torch.exp, "sigmoid": torch.sigmoid, "tanh": torch.tanh,
       "abs": torch.abs, "neg": torch.neg, "gelu": F.gelu, "silu": F.silu, "hardswish": F.hardswish,
       "leaky_relu": F.leaky_relu, "mul2": lambda a: a * 2, "add_self": lambda a: a + a,
       "clone": torch.clone, "to_bf16": lambda a: a.to(torch.bfloat16), "sqrt_abs": lambda a: a.abs().sqrt(),
       "relu6": F.relu6, "elu": F.elu, "softplus": F.softplus, "dropout_eval": lambda a: F.dropout(a, 0.0)}
bad = []
for n, f in ops.items():
  c = f(x).stride()
  t = f(x.to("tpu")).stride()
  if c != t:
    bad.append(n)
  print(f"{'MISMATCH' if c != t else 'ok':8s} {n:12s} cpu={c} tpu={t}")
print("not preserved on TPU:", bad)
