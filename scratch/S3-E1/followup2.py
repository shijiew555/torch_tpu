import sys
import traceback

import torch
import torch_tpu  # noqa: F401
from torch import nn

torch.set_float32_matmul_precision("highest")
T = "tpu"
which = sys.argv[1]


def run(name, f):
  try:
    print(f"== {name}: {f()}", flush=True)
  except Exception as e:  # noqa: BLE001
    print(f"== {name}: FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)


if which == "amp":
  def unscale():
    g = [torch.tensor([1.0, float("inf")], device=T)]
    fi = torch.zeros(1, device=T)
    torch._amp_foreach_non_finite_check_and_unscale_(
        g, fi, torch.ones(1, device=T))
    return fi.cpu()

  def update():
    s = torch.full((1,), 1024.0, device=T)
    gt = torch.zeros(1, dtype=torch.int32, device=T)
    torch._amp_update_scale_(s, gt, torch.ones(1, device=T), 2.0, 0.5, 2000)
    return s.cpu()

  run("unscale", unscale)
  run("update_scale", update)
  run("GradScaler('tpu').is_enabled",
      lambda: torch.amp.GradScaler("tpu").is_enabled())
  run("GradScaler() default device", lambda: torch.amp.GradScaler()._device)

if which == "dist":
  D = torch.distributions
  for name, f in [
      ("MVN", lambda: D.MultivariateNormal(torch.zeros(3, device=T),
                                           torch.eye(3, device=T)).sample()),
      ("Poisson", lambda: D.Poisson(torch.full((3,), 4.0, device=T)).sample()),
      ("Uniform", lambda: D.Uniform(torch.zeros(3, device=T),
                                    torch.ones(3, device=T)).rsample()),
      ("Exponential", lambda: D.Exponential(torch.ones(3, device=T)).sample()),
      ("Laplace", lambda: D.Laplace(torch.zeros(3, device=T), 1.0).sample()),
      ("StudentT", lambda: D.StudentT(torch.full((3,), 3.0, device=T)).sample()),
      ("Binomial", lambda: D.Binomial(10, torch.full((3,), 0.3, device=T)).sample()),
      ("Multinomial", lambda: D.Multinomial(
          10, torch.tensor([0.2, 0.8], device=T)).sample()),
      ("OneHotCategorical", lambda: D.OneHotCategorical(
          torch.tensor([0.2, 0.8], device=T)).sample()),
      ("RelaxedBernoulli", lambda: D.RelaxedBernoulli(
          torch.tensor(0.5, device=T), probs=torch.full((3,), 0.3, device=T)).rsample()),
      ("Gumbel", lambda: nn.functional.gumbel_softmax(
          torch.zeros(2, 3, device=T), hard=True)),
      ("kl Normal", lambda: D.kl_divergence(
          D.Normal(torch.zeros(3, device=T), 1.0),
          D.Normal(torch.ones(3, device=T), 2.0))),
      ("Gamma", lambda: D.Gamma(torch.full((3,), 2.0, device=T), 1.0).sample()),
      ("Chi2", lambda: D.Chi2(torch.full((3,), 2.0, device=T)).sample()),
      ("Beta", lambda: D.Beta(torch.full((3,), 2.0, device=T), 3.0).sample()),
  ]:
    run(name, lambda f=f: f().cpu())

if which == "bn":
  for dev in ("cpu", T):
    bn = nn.BatchNorm1d(4, momentum=None).to(dev)
    torch.manual_seed(0)
    x = (torch.randn(16, 4) + 5).to(dev)
    bn(x)
    bn(x * 2)
    print(dev, "mom None mean", bn.running_mean.cpu(), "batches",
          bn.num_batches_tracked.cpu())

if which == "e2e":
  import torch.utils.checkpoint as cp

  class Net(nn.Module):
    def __init__(self, ckpt):
      super().__init__()
      self.ckpt = ckpt
      self.layers = nn.ModuleList(
          nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True)
          for _ in range(2))
      self.head = nn.Linear(32, 1)

    def forward(self, x):
      for l in self.layers:
        x = cp.checkpoint(l, x, use_reentrant=False) if self.ckpt else l(x)
      return self.head(x).mean()

  for dev in ("cpu", T):
    grads = []
    for ckpt in (False, True):
      torch.manual_seed(0)
      net = Net(ckpt).to(dev).train()
      x = torch.randn(2, 8, 32).to(dev)
      torch.manual_seed(42)
      net(x).backward()
      grads.append(net.layers[0].linear1.weight.grad.cpu())
    rel = ((grads[0] - grads[1]).norm() / grads[0].norm()).item()
    print(f"{dev}: TransformerEncoderLayer(dropout=0.1) grad rel diff, "
          f"checkpoint vs none = {rel:.3g}")

if which == "compiled":
  import torch.utils.checkpoint as cp
  lin = nn.Linear(8, 8).to(T)
  f = lambda x: nn.functional.dropout(lin(x), 0.5)
  g = torch.compile(lambda x: cp.checkpoint(f, x, use_reentrant=False),
                    backend="tpu")
  x = torch.randn(4, 8, device=T, requires_grad=True)
  torch.manual_seed(1)
  y = g(x)
  y.sum().backward()
  # In-graph check: grad of dropout(lin(x)) wrt x = mask*2 @ W.
  mask = (y != 0).float() * 2
  exp = mask @ lin.weight
  print("compiled checkpoint grad matches fwd mask:",
        (x.grad - exp).abs().max().item())
