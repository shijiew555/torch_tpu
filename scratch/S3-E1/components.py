"""Training loop component checks on TPU vs CPU."""
import faulthandler
import io
import sys
import traceback

import torch
import torch_tpu  # noqa: F401
from torch import nn

faulthandler.dump_traceback_later(900, exit=True)
torch.set_float32_matmul_precision("highest")
T = "tpu"
TESTS = []


def test(f):
  TESTS.append(f)
  return f


def close(a, b, tol=1e-4):
  if isinstance(a, (list, tuple)):
    return max(close(x, y, tol) for x, y in zip(a, b))
  d = (a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item()
  return d


def model(dev, seed=0):
  torch.manual_seed(seed)
  return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(dev)


@test
def adam_capturable():
  m = model(T)
  opt = torch.optim.Adam(m.parameters(), lr=0.01, capturable=True)
  for _ in range(3):
    m(torch.ones(4, 8, device=T)).sum().backward()
    opt.step()
  mc = model("cpu")
  optc = torch.optim.Adam(mc.parameters(), lr=0.01)
  for _ in range(3):
    mc(torch.ones(4, 8)).sum().backward()
    optc.step()
  return close(list(m.parameters()), list(mc.parameters()))


@test
def optim_state_dict_roundtrip():
  m = model(T)
  opt = torch.optim.AdamW(m.parameters(), lr=0.01)
  x = torch.randn(4, 8).to(T)
  for _ in range(2):
    opt.zero_grad()
    m(x).sum().backward()
    opt.step()
  buf = io.BytesIO()
  torch.save({"m": m.state_dict(), "o": opt.state_dict()}, buf)
  buf.seek(0)
  sd_cpu = torch.load(buf, map_location="cpu")
  buf.seek(0)
  sd_tpu = torch.load(buf, map_location=T)
  st = sd_tpu["o"]["state"][0]
  info = {k: (v.device, v.dtype) for k, v in st.items()}
  # Resume on CPU from TPU checkpoint and on TPU from CPU checkpoint.
  m2 = model("cpu")
  m2.load_state_dict(sd_cpu["m"])
  o2 = torch.optim.AdamW(m2.parameters(), lr=0.01)
  o2.load_state_dict(sd_cpu["o"])
  m3 = model(T)
  m3.load_state_dict(sd_cpu["m"])
  o3 = torch.optim.AdamW(m3.parameters(), lr=0.01)
  o3.load_state_dict(sd_cpu["o"])
  for mm, oo, xx in ((m, opt, x), (m2, o2, x.cpu()), (m3, o3, x)):
    oo.zero_grad()
    mm(xx).sum().backward()
    oo.step()
  s3 = o3.state[next(iter(m3.parameters()))]
  return (close(list(m.parameters()), list(m2.parameters())),
          close(list(m.parameters()), list(m3.parameters())),
          info, {k: v.device for k, v in s3.items()})


@test
def zero_grad_not_none():
  m = model(T)
  m(torch.ones(2, 8, device=T)).sum().backward()
  opt = torch.optim.SGD(m.parameters(), lr=0.1)
  opt.zero_grad(set_to_none=False)
  return [p.grad.abs().sum().item() for p in m.parameters()]


@test
def schedulers():
  res = {}
  for dev in ("cpu", T):
    m = model(dev)
    out = []
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    s1 = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.0, total_steps=10)
    for _ in range(5):
      opt.step()
      s1.step()
      out.append(opt.param_groups[0]["lr"])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    a = torch.optim.lr_scheduler.LinearLR(opt, 0.1, total_iters=2)
    b = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    s = torch.optim.lr_scheduler.SequentialLR(opt, [a, b], milestones=[2])
    for _ in range(5):
      opt.step()
      s.step()
      out.append(opt.param_groups[0]["lr"])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    r = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0)
    for _ in range(4):
      loss = torch.tensor(1.0, device=dev)
      r.step(loss)
      out.append(opt.param_groups[0]["lr"])
    res[dev] = out
  return res["cpu"] == res[T], res[T]


@test
def clip_grad():
  out = []
  for foreach in (False, True, None):
    gs = []
    for dev in ("cpu", T):
      m = model(dev)
      m(torch.ones(4, 8, device=dev) * 3).sum().backward()
      n = nn.utils.clip_grad_norm_(m.parameters(), 0.5, foreach=foreach)
      gs.append((n, [p.grad for p in m.parameters()]))
    out.append((close(gs[0][0], gs[1][0]), close(gs[0][1], gs[1][1])))
  for norm_type in (1.0, float("inf"), 3.0):
    gs = []
    for dev in ("cpu", T):
      m = model(dev)
      m(torch.ones(4, 8, device=dev) * 3).sum().backward()
      gs.append(nn.utils.clip_grad_norm_(m.parameters(), 0.5,
                                         norm_type=norm_type))
    out.append(("norm", norm_type, close(gs[0], gs[1])))
  # nonfinite
  m = model(T)
  m(torch.ones(4, 8, device=T)).sum().backward()
  next(m.parameters()).grad[0, 0] = float("nan")
  try:
    nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
    out.append("nan: no error")
  except RuntimeError as e:
    out.append("nan raised: " + str(e)[:60])
  m = model(T)
  m(torch.ones(4, 8, device=T)).sum().backward()
  next(m.parameters()).grad[0, 0] = float("inf")
  n = nn.utils.clip_grad_norm_(m.parameters(), 1.0)
  out.append(("inf norm", n.item()))
  m = model(T)
  m(torch.ones(4, 8, device=T)).sum().backward()
  nn.utils.clip_grad_value_(m.parameters(), 0.01)
  out.append(("clip_value max", max(p.grad.abs().max().item()
                                    for p in m.parameters())))
  g = [p.grad for p in m.parameters()]
  out.append(("total_norm", nn.utils.get_total_norm(g).item(),
              nn.utils.get_total_norm([x.cpu() for x in g]).item()))
  return out


@test
def gradscaler_tpu():
  out = []
  m = model(T)
  opt = torch.optim.SGD(m.parameters(), lr=0.1)
  sc = torch.amp.GradScaler("tpu", init_scale=1024.0)
  before = [p.detach().clone() for p in m.parameters()]
  with torch.autocast("tpu", dtype=torch.float16):
    loss = m(torch.ones(4, 8, device=T)).sum()
  sc.scale(loss).backward()
  next(m.parameters()).grad[0, 0] = float("inf")
  sc.unscale_(opt)
  sc.step(opt)
  sc.update()
  out.append(("skipped", close(before, list(m.parameters())) == 0,
              "scale", sc.get_scale()))
  for _ in range(3):
    opt.zero_grad()
    with torch.autocast("tpu", dtype=torch.float16):
      loss = m(torch.ones(4, 8, device=T)).sum()
    sc.scale(loss).backward()
    sc.step(opt)
    sc.update()
  out.append(("after", sc.get_scale(), loss.item()))
  return out


@test
def gradscaler_default_device():
  m = model(T)
  opt = torch.optim.SGD(m.parameters(), lr=0.1)
  sc = torch.amp.GradScaler()
  loss = m(torch.ones(4, 8, device=T)).sum()
  sc.scale(loss).backward()
  sc.step(opt)
  sc.update()
  return sc.get_scale(), sc._enabled


@test
def gradscaler_legacy_cuda_api():
  # Many scripts use torch.cuda.amp.GradScaler; just report behavior.
  sc = torch.tpu.amp.GradScaler() if hasattr(torch.tpu, "amp") else None
  return sc


@test
def checkpoint_dropout():
  import torch.utils.checkpoint as cp
  out = []
  for reentrant in (True, False):
    torch.manual_seed(0)
    lin = nn.Linear(8, 8).to(T)
    drop = nn.Dropout(0.5)

    def f(x):
      return drop(lin(x)).relu()

    x = torch.randn(4, 8, device=T, requires_grad=True)
    torch.manual_seed(1)
    y = cp.checkpoint(f, x, use_reentrant=reentrant)
    y.sum().backward()
    g1 = x.grad.clone()
    x.grad = None
    torch.manual_seed(1)
    y2 = f(x)
    y2.sum().backward()
    out.append((reentrant, close(y, y2), close(g1, x.grad)))
  return out


@test
def func_transforms():
  from torch.func import functional_call, grad, jacrev, vmap
  out = []
  m = nn.Linear(4, 3).to(T)
  mc = nn.Linear(4, 3)
  mc.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
  x = torch.randn(5, 4)
  p = dict(m.named_parameters())
  pc = dict(mc.named_parameters())

  def loss(params, xx):
    return functional_call(m if xx.device.type == T else mc, params,
                           (xx,)).pow(2).sum()

  g = grad(loss)(p, x.to(T))
  gc = grad(loss)(pc, x)
  out.append(("grad", close(list(g.values()), list(gc.values()))))
  ps = vmap(grad(loss), in_dims=(None, 0))(p, x.to(T).unsqueeze(1))
  psc = vmap(grad(loss), in_dims=(None, 0))(pc, x.unsqueeze(1))
  out.append(("per-sample", close(list(ps.values()), list(psc.values()))))
  j = jacrev(lambda xx: torch.sin(xx) @ xx.T)(x.to(T)[:3])
  jc = jacrev(lambda xx: torch.sin(xx) @ xx.T)(x[:3])
  out.append(("jacrev", close(j, jc)))
  v = vmap(torch.dot)(x.to(T), x.to(T))
  out.append(("vmap dot", close(v, (x * x).sum(1))))
  from torch.func import hessian, jvp
  h = hessian(lambda xx: xx.sin().sum())(x.to(T)[0])
  out.append(("hessian", close(h, torch.diag(-x[0].sin()))))
  _, t = jvp(torch.sin, (x.to(T),), (torch.ones(5, 4, device=T),))
  out.append(("jvp", close(t, x.cos())))
  return out


@test
def parametrizations():
  from torch.nn.utils import parametrizations as P
  out = []
  for name, fn in (("weight_norm", P.weight_norm), ("orthogonal", P.orthogonal),
                   ("orthogonal_cayley", lambda l: P.orthogonal(l, orthogonal_map="cayley")),
                   ("orthogonal_householder", lambda l: P.orthogonal(l, orthogonal_map="householder")),
                   ("spectral_norm", P.spectral_norm)):
    res = []
    try:
     for dev in ("cpu", T):
      torch.manual_seed(0)
      lin = nn.Linear(6, 6)
      lin = fn(lin).to(dev)
      opt = torch.optim.SGD(lin.parameters(), lr=0.1)
      torch.manual_seed(0)
      x = torch.randn(3, 6).to(dev)
      for _ in range(3):
        opt.zero_grad()
        lin(x).pow(2).sum().backward()
        opt.step()
      res.append(lin.weight)
     out.append((name, close(res[0], res[1])))
    except Exception as e:  # noqa: BLE001
      out.append((name, "FAIL", str(e)[:120]))
  return out


@test
def distributions():
  D = torch.distributions
  out = []
  mk = [
      ("Normal", lambda d: D.Normal(torch.zeros(3, device=d),
                                    torch.ones(3, device=d))),
      ("Categorical", lambda d: D.Categorical(
          logits=torch.tensor([0.1, 0.5, 2.0], device=d))),
      ("Bernoulli", lambda d: D.Bernoulli(probs=torch.full((3,), 0.3,
                                                           device=d))),
      ("Beta", lambda d: D.Beta(torch.full((3,), 2.0, device=d),
                                torch.full((3,), 3.0, device=d))),
      ("Dirichlet", lambda d: D.Dirichlet(torch.tensor([1.0, 2.0, 3.0],
                                                       device=d))),
      ("Gamma", lambda d: D.Gamma(torch.full((3,), 2.0, device=d),
                                  torch.full((3,), 1.0, device=d))),
      ("MVN", lambda d: D.MultivariateNormal(torch.zeros(3, device=d),
                                             torch.eye(3, device=d))),
      ("Poisson", lambda d: D.Poisson(torch.full((3,), 4.0, device=d))),
      ("Uniform", lambda d: D.Uniform(torch.zeros(3, device=d),
                                      torch.ones(3, device=d))),
  ]
  for name, f in mk:
    try:
      dt, dc = f(T), f("cpu")
      s = dt.sample((20000,))
      sc = dc.sample((20000,))
      mean_d = (s.float().mean(0).cpu() - sc.float().mean(0)).abs().max().item()
      lp = close(dt.log_prob(s[:5]), dc.log_prob(s[:5].cpu()))
      ent = None
      try:
        ent = close(dt.entropy(), dc.entropy())
      except NotImplementedError:
        pass
      r = ""
      if dt.has_rsample:
        rs = dt.rsample((4,))
        r = f" rsample dev={rs.device}"
      out.append(f"{name}: meandiff={mean_d:.3g} lp={lp:.3g} ent={ent}{r}")
    except Exception as e:  # noqa: BLE001
      out.append(f"{name}: FAIL {type(e).__name__}: {str(e)[:200]}")
  # rsample gradient
  try:
    mu = torch.zeros(3, device=T, requires_grad=True)
    D.Normal(mu, 1.0).rsample().sum().backward()
    a = torch.full((3,), 2.0, device=T, requires_grad=True)
    D.Gamma(a, 1.0).rsample().sum().backward()
    out.append(f"rsample grads ok: {mu.grad.cpu()} {a.grad.cpu()}")
  except Exception as e:  # noqa: BLE001
    out.append(f"rsample grad FAIL {type(e).__name__}: {str(e)[:200]}")
  return out


@test
def swa_ema():
  from torch.optim.swa_utils import AveragedModel, SWALR, get_ema_multi_avg_fn
  out = []
  for dev in ("cpu", T):
    m = model(dev)
    swa = AveragedModel(m)
    ema = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sch = SWALR(opt, swa_lr=0.05)
    for _ in range(3):
      opt.zero_grad()
      m(torch.ones(4, 8, device=dev)).sum().backward()
      opt.step()
      swa.update_parameters(m)
      ema.update_parameters(m)
      sch.step()
    out.append((list(swa.parameters()), list(ema.parameters())))
  ps = [p.detach().clone() for p in model(T).parameters()]
  qs = [p.detach().clone() + 1 for p in ps]
  torch._foreach_lerp_(ps, qs, 0.1)
  torch._foreach_mul_(ps, 0.5)
  torch._foreach_add_(ps, qs, alpha=0.5)
  pc = [p.detach().clone() for p in model("cpu").parameters()]
  qc = [p + 1 for p in pc]
  torch._foreach_lerp_(pc, qc, 0.1)
  torch._foreach_mul_(pc, 0.5)
  torch._foreach_add_(pc, qc, alpha=0.5)
  return (close(out[0][0], out[1][0]), close(out[0][1], out[1][1]),
          close(ps, pc))


@test
def update_bn_swa():
  from torch.optim.swa_utils import AveragedModel, update_bn
  m = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8)).to(T)
  swa = AveragedModel(m)
  loader = [torch.randn(4, 8, device=T) for _ in range(3)]
  update_bn(loader, swa)
  return swa.module[1].running_mean.cpu()


only = sys.argv[1:]
for t in TESTS:
  if only and t.__name__ not in only:
    continue
  try:
    print(f"== {t.__name__}: {t()}", flush=True)
  except Exception as e:  # noqa: BLE001
    print(f"== {t.__name__}: FAIL {type(e).__name__}: {str(e)[:400]}",
          flush=True)
    traceback.print_exc(limit=-4)
