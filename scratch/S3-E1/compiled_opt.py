import faulthandler, sys, traceback
import torch, torch_tpu  # noqa: F401
from torch import nn
faulthandler.dump_traceback_later(600, exit=True)
torch.set_float32_matmul_precision("highest")

def mk(dev):
  torch.manual_seed(0)
  return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(dev)

def train(dev, cls, kw, compile_step, clip):
  m = mk(dev)
  opt = cls(m.parameters(), **kw)
  def step():
    if clip:
      nn.utils.clip_grad_norm_(m.parameters(), 0.1)
    opt.step()
  if compile_step:
    step = torch.compile(step, backend="tpu")
  torch.manual_seed(1)
  for _ in range(4):
    x = torch.randn(4, 8).to(dev)
    opt.zero_grad()
    m(x).pow(2).sum().backward()
    step()
  return [p.detach().cpu() for p in m.parameters()]

for cls, kw in [(torch.optim.Adam, dict(lr=0.01)),
                (torch.optim.AdamW, dict(lr=torch.tensor(0.01), weight_decay=0.1)),
                (torch.optim.SGD, dict(lr=0.1, momentum=0.9)),
                (torch.optim.Adam, dict(lr=0.01, capturable=True))]:
  for clip in (False, True):
    name = f"{cls.__name__} {[k for k in kw]} clip={clip}"
    try:
      kwc = dict(kw); kwc.pop("capturable", None)
      ref = train("cpu", cls, kwc, False, clip)
      out = train("tpu", cls, kw, True, clip)
      print(name, "compiled step maxdiff",
            max((a - b).abs().max().item() for a, b in zip(out, ref)), flush=True)
    except Exception as e:
      print(name, "FAIL", type(e).__name__, str(e)[:300], flush=True)

# Whole compiled training step.
def full(dev, comp):
  m = mk(dev); opt = torch.optim.AdamW(m.parameters(), lr=0.01)
  def ts(x):
    opt.zero_grad(set_to_none=True)
    loss = m(x).pow(2).sum(); loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); return loss
  if comp: ts = torch.compile(ts, backend="tpu")
  torch.manual_seed(1)
  ls = [ts(torch.randn(4, 8).to(dev)).item() for _ in range(4)]
  return ls, [p.detach().cpu() for p in m.parameters()]
try:
  a = full("cpu", False); b = full("tpu", True)
  print("full compiled step: losses", a[0], b[0], "maxdiff",
        max((x - y).abs().max().item() for x, y in zip(a[1], b[1])))
except Exception as e:
  print("full compiled FAIL", type(e).__name__, str(e)[:300]); traceback.print_exc(limit=-3)
