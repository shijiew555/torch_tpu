import io
import sys

import torch
import torch_tpu  # noqa: F401
from torch import nn

torch.set_float32_matmul_precision("highest")
T = "tpu"
which = sys.argv[1]

if which == "ckpt":
  import torch.utils.checkpoint as cp
  for dev in ("cpu", T):
    for reentrant in (True, False):
      torch.manual_seed(0)
      lin = nn.Linear(8, 8).to(dev)
      drop = nn.Dropout(0.5)
      f = lambda x: drop(lin(x))
      x = torch.randn(4, 8, device=dev, requires_grad=True)
      torch.manual_seed(1)
      y = cp.checkpoint(f, x, use_reentrant=reentrant)
      y.sum().backward()
      g1 = x.grad.clone()
      x.grad = None
      torch.manual_seed(1)
      y2 = f(x)
      y2.sum().backward()
      print(dev, "reentrant", reentrant, "fwd diff",
            (y - y2).abs().max().item(), "grad diff",
            (g1 - x.grad).abs().max().item())
  # direct rng state round trip
  torch.manual_seed(3)
  s = torch.tpu.get_rng_state()
  a = torch.rand(4, device=T)
  torch.tpu.set_rng_state(s)
  b = torch.rand(4, device=T)
  print("tpu get/set_rng_state replay diff", (a - b).abs().max().item())
  from torch.utils.checkpoint import _get_device_module, _infer_device_type
  print("device type inferred", _infer_device_type(torch.ones(1, device=T)))
  dm = _get_device_module(T)
  print("device module", dm, hasattr(dm, "get_rng_state"))
  with torch.random.fork_rng(devices=[0], device_type=T):
    c = torch.rand(4, device=T)
  d = torch.rand(4, device=T)
  print("fork_rng: rand inside == rand after?", torch.equal(c, d))

if which == "bn":
  for dev in ("cpu", T):
    bn = nn.BatchNorm1d(4, momentum=None).to(dev)
    torch.manual_seed(0)
    x = (torch.randn(16, 4) + 5).to(dev)
    bn(x)
    bn(x * 2)
    print(dev, "momentum=None running_mean", bn.running_mean.cpu(),
          "num_batches", bn.num_batches_tracked.cpu())
    bn = nn.BatchNorm1d(4).to(dev)
    bn(x)
    print(dev, "momentum=0.1 running_mean", bn.running_mean.cpu())

if which == "sd":
  def mk(dev):
    torch.manual_seed(0)
    return nn.Linear(8, 2).to(dev)
  x = torch.randn(4, 8)
  m = mk(T)
  opt = torch.optim.AdamW(m.parameters(), lr=0.01)
  for _ in range(2):
    opt.zero_grad(); m(x.to(T)).sum().backward(); opt.step()
  buf = io.BytesIO()
  torch.save({"m": m.state_dict(), "o": opt.state_dict()}, buf)
  buf.seek(0)
  sd = torch.load(buf, map_location="cpu")
  m3 = mk(T); m3.load_state_dict(sd["m"])
  o3 = torch.optim.AdamW(m3.parameters(), lr=0.01); o3.load_state_dict(sd["o"])
  print("param diff after load", max((a - b).abs().max().item()
        for a, b in zip(m.parameters(), m3.parameters())))
  for k in ("exp_avg", "exp_avg_sq", "step"):
    a = opt.state[m.weight][k]; b = o3.state[m3.weight][k]
    print(k, a.device, b.device, (a.cpu() - b.cpu()).abs().max().item())
  for mm, oo in ((m, opt), (m3, o3)):
    oo.zero_grad(); mm(x.to(T)).sum().backward()
  print("grad diff", (m.weight.grad - m3.weight.grad).abs().max().item())
  opt.step(); o3.step()
  print("param diff after step", max((a - b).abs().max().item()
        for a, b in zip(m.parameters(), m3.parameters())))
  print("opt step dev", opt.state[m.weight]["step"].device,
        "o3 step dev", o3.state[m3.weight]["step"].device)
