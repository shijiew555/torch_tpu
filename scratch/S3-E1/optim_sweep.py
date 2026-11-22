"""Run optimizers for 4 steps on TPU and CPU, compare params."""
import copy
import faulthandler
import sys
import traceback

import torch
import torch_tpu  # noqa: F401

faulthandler.dump_traceback_later(600, exit=True)
torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")


def make_model(dtype):
  torch.manual_seed(0)
  return torch.nn.Sequential(
      torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)
  ).to(dtype)


X = torch.randn(8, 16)
Y = torch.randn(8, 4)

O = torch.optim
CASES = [
    ("SGD", O.SGD, dict(lr=0.1)),
    ("SGD mom nesterov wd", O.SGD,
     dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01)),
    ("SGD foreach=False mom", O.SGD, dict(lr=0.1, momentum=0.9, foreach=False)),
    ("SGD fused", O.SGD, dict(lr=0.1, momentum=0.9, fused=True)),
    ("Adam", O.Adam, dict(lr=0.01)),
    ("Adam foreach=False", O.Adam, dict(lr=0.01, foreach=False)),
    ("Adam fused", O.Adam, dict(lr=0.01, fused=True)),
    ("Adam amsgrad", O.Adam, dict(lr=0.01, amsgrad=True)),
    ("Adam capturable", O.Adam, dict(lr=0.01, capturable=True)),
    ("Adam maximize", O.Adam, dict(lr=0.01, maximize=True)),
    ("Adam tensor lr", O.Adam, dict(lr=torch.tensor(0.01))),
    ("AdamW", O.AdamW, dict(lr=0.01, weight_decay=0.1)),
    ("AdamW fused amsgrad", O.AdamW, dict(lr=0.01, fused=True, amsgrad=True)),
    ("AdamW foreach=False amsgrad", O.AdamW,
     dict(lr=0.01, foreach=False, amsgrad=True)),
    ("Adagrad", O.Adagrad, dict(lr=0.1)),
    ("Adagrad fused", O.Adagrad, dict(lr=0.1, fused=True)),
    ("RMSprop", O.RMSprop, dict(lr=0.01, momentum=0.9, centered=True)),
    ("Adafactor", O.Adafactor, dict(lr=0.01)),
    ("Adafactor foreach=False", O.Adafactor, dict(lr=0.01, foreach=False)),
    ("NAdam", O.NAdam, dict(lr=0.01)),
    ("NAdam decoupled", O.NAdam, dict(lr=0.01, decoupled_weight_decay=True,
                                      weight_decay=0.1)),
    ("RAdam", O.RAdam, dict(lr=0.01)),
    ("Adamax", O.Adamax, dict(lr=0.01)),
    ("ASGD", O.ASGD, dict(lr=0.01)),
    ("Rprop", O.Rprop, dict(lr=0.01)),
    ("Adadelta", O.Adadelta, dict(lr=1.0)),
    ("SparseAdam-skip", None, None),
    ("LBFGS", O.LBFGS, dict(lr=0.5, max_iter=5)),
    ("LBFGS strong_wolfe", O.LBFGS,
     dict(lr=1, max_iter=5, line_search_fn="strong_wolfe")),
]


def run(device, cls, kw, dtype, steps=4):
  m = make_model(dtype).to(device)
  kw = dict(kw)
  if isinstance(kw.get("lr"), torch.Tensor):
    kw["lr"] = kw["lr"].clone()
  opt = cls(m.parameters(), **kw)
  x = X.to(device, dtype)
  y = Y.to(device, dtype)
  for _ in range(steps):
    def closure():
      opt.zero_grad()
      loss = torch.nn.functional.mse_loss(m(x), y)
      loss.backward()
      return loss
    if cls is O.LBFGS:
      opt.step(closure)
    else:
      closure()
      opt.step()
  return [p.detach().float().cpu() for p in m.parameters()]


only = sys.argv[1:] if len(sys.argv) > 1 else None
for dtype in (torch.float32, torch.bfloat16):
  for name, cls, kw in CASES:
    if cls is None or (only and not any(o in name for o in only)):
      continue
    tag = f"{name} [{str(dtype)[6:]}]"
    try:
      ref = run("cpu", cls, kw, dtype)
    except Exception as e:  # noqa: BLE001
      print(f"CPUFAIL {tag}: {type(e).__name__}: {str(e)[:150]}")
      continue
    try:
      out = run("tpu", cls, kw, dtype)
      d = max((a - b).abs().max().item() for a, b in zip(out, ref))
      tol = 1e-3 if dtype == torch.float32 else 3e-2
      st = "OK " if d < tol else "DIFF"
      print(f"{st} {tag}: maxdiff={d:.3g}", flush=True)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL {tag}: {type(e).__name__}: {str(e)[:300]}", flush=True)
      traceback.print_exc(limit=-3)
