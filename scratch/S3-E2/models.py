"""torchvision model sweep: forward, backward, SGD step on TPU vs CPU.

Usage: python models.py [--compile] name[:size] ...
"""
import copy
import faulthandler
import sys
import time
import traceback

import torch
import torch_tpu  # noqa: F401
import torchvision.models as tvm
import torchvision.models.detection as tvd
import torchvision.models.segmentation as tvs
import torchvision.models.video as tvv

torch.set_float32_matmul_precision("highest")
faulthandler.dump_traceback_later(1500, exit=True)

COMPILE = "--compile" in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith("--")]
DEV = torch.device("tpu")


def build(name):
  for mod in (tvm, tvs, tvv, tvd):
    if hasattr(mod, name):
      fn = getattr(mod, name)
      break
  kw = {"weights": None}
  if name in ("googlenet", "inception_v3"):
    kw["aux_logits"] = True
    kw["init_weights"] = True
  if name in ("deeplabv3_resnet50", "fcn_resnet50", "lraspp_mobilenet_v3_large"):
    kw["weights_backbone"] = None
    kw["num_classes"] = 5
  if hasattr(tvd, name):
    kw["weights_backbone"] = None
    kw["num_classes"] = 3
  return fn(**kw)


def to_loss(out):
  if isinstance(out, torch.Tensor):
    return out.float().pow(2).mean(), out
  if isinstance(out, dict):
    loss = sum(v.float().pow(2).mean() if v.dim() else v for v in out.values())
    return loss, out
  if isinstance(out, (tuple, list)) and hasattr(out, "_fields"):
    return sum(o.float().pow(2).mean() for o in out if o is not None), out
  if isinstance(out, (tuple, list)):
    return sum(o.float().pow(2).mean() for o in out if o is not None), out
  raise TypeError(type(out))


def run(model, x, targets=None):
  out = model(x, targets) if targets is not None else model(x)
  loss, out = to_loss(out)
  loss.backward()
  return loss


def rel(a, b):
  a = a.detach().float().cpu()
  b = b.detach().float().cpu()
  return ((a - b).norm() / (b.norm() + 1e-12)).item()


for spec in names:
  name, _, size = spec.partition(":")
  size = int(size or 64)
  torch.manual_seed(0)
  try:
    m_cpu = build(name)
    m_cpu.train()
    for mod in m_cpu.modules():
      if isinstance(mod, torch.nn.Dropout) or type(mod).__name__ == "StochasticDepth":
        mod.p = 0.0
    det = hasattr(tvd, name)
    vid = hasattr(tvv, name)
    if vid:
      x = torch.randn(2, 3, 8, size, size)
    else:
      x = torch.randn(2, 3, size, size)
    targets = None
    if det:
      targets = [
          {"boxes": torch.tensor([[4.0, 4.0, 40.0, 40.0]]),
           "labels": torch.tensor([1]),
           "masks": torch.zeros(1, size, size, dtype=torch.uint8)},
          {"boxes": torch.tensor([[10.0, 8.0, 50.0, 60.0]]),
           "labels": torch.tensor([2]),
           "masks": torch.zeros(1, size, size, dtype=torch.uint8)},
      ]
      x = [x[0], x[1]]
    m_tpu = copy.deepcopy(m_cpu).to(DEV)
    if det:
      x_t = [i.to(DEV) for i in x]
      t_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
    else:
      x_t = x.to(DEV)
      t_t = None
    f_tpu = torch.compile(m_tpu, backend="tpu") if COMPILE else m_tpu
    # Detection models are random; force the same RNG.
    torch.manual_seed(1)
    l_cpu = run(m_cpu, x, targets)
    t0 = time.time()
    torch.manual_seed(1)
    l_tpu = run(f_tpu, x_t, t_t)
    l_tpu_v = l_tpu.item()
    t1 = time.time()
    worst = ("", 0.0)
    pairs = [(n, p, q) for (n, p), (_, q) in zip(m_cpu.named_parameters(), m_tpu.named_parameters())
             if p.grad is not None or q.grad is not None]
    gscale = max(p.grad.norm().item() for _, p, _ in pairs if p.grad is not None)
    tot_d = 0.0
    tot_n = 0.0
    for n, p, q in pairs:
      if (p.grad is None) != (q.grad is None):
        worst = (n + " grad None mismatch", 1.0)
        break
      d = (q.grad.detach().cpu().float() - p.grad.float()).norm().item()
      tot_d += d * d
      tot_n += p.grad.norm().item() ** 2
      r = d / (p.grad.norm().item() + 1e-3 * gscale)
      if r > worst[1]:
        worst = (n, r)
    worst = (worst[0], worst[1])
    total_rel = (tot_d / max(tot_n, 1e-30)) ** 0.5
    bworst = ("", 0.0)
    for (n, b), (_, c) in zip(m_cpu.named_buffers(), m_tpu.named_buffers()):
      if b.is_floating_point():
        r = rel(c, b)
        if r > bworst[1]:
          bworst = (n, r)
    sgd_c = torch.optim.SGD(m_cpu.parameters(), lr=0.01)
    sgd_t = torch.optim.SGD(m_tpu.parameters(), lr=0.01)
    sgd_c.step()
    sgd_t.step()
    torch.manual_seed(1)
    with torch.no_grad():
      m_cpu.eval()
      m_tpu.eval()
      o2c = m_cpu(x)
      o2t = (torch.compile(m_tpu, backend="tpu") if COMPILE else m_tpu)(x_t)
      if det:
        o2 = f"eval boxes cpu={o2c[0]['boxes'].shape[0]} tpu={o2t[0]['boxes'].shape[0]}"
      else:
        lc, _ = to_loss(o2c)
        lt, _ = to_loss(o2t)
        o2 = f"eval-after-step rel={abs(lt.item() - lc.item()) / (abs(lc.item()) + 1e-12):.2e}"
    print(f"OK {name} size={size} compile={COMPILE} loss cpu={l_cpu.item():.6g} "
          f"tpu={l_tpu_v:.6g} grad_total_rel={total_rel:.2e} worst_grad={worst[0]} rel={worst[1]:.2e} "
          f"worst_buf={bworst[0]} rel={bworst[1]:.2e} {o2} "
          f"tpu_time={t1 - t0:.1f}s", flush=True)
  except Exception as e:  # noqa: BLE001
    msg = traceback.format_exc().strip().splitlines()
    print(f"FAIL {name} compile={COMPILE}: {type(e).__name__}: "
          f"{str(e).splitlines()[0][:300] if str(e) else ''}", flush=True)
    print("   " + "\n   ".join(msg[-8:]), flush=True)
