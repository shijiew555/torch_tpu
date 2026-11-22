import faulthandler, traceback
import torch, torch_tpu  # noqa: F401
from torch import nn
faulthandler.dump_traceback_later(600, exit=True)
torch.set_float32_matmul_precision("highest")
T = "tpu"

def run(name, f):
  try:
    print(f"== {name}: {f()}", flush=True)
  except Exception as e:
    print(f"== {name}: FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)

def mixed_dtype_foreach():
  res = []
  for dev in ("cpu", T):
    torch.manual_seed(0)
    a = nn.Linear(4, 4).to(dev); b = nn.Linear(4, 4).to(dev).bfloat16()
    opt = torch.optim.AdamW(list(a.parameters()) + list(b.parameters()), lr=0.1, foreach=True)
    for _ in range(3):
      opt.zero_grad()
      (a(torch.ones(2, 4, device=dev)).sum() + b(torch.ones(2, 4, device=dev, dtype=torch.bfloat16)).float().sum()).backward()
      nn.utils.clip_grad_norm_(list(a.parameters()) + list(b.parameters()), 0.5)
      opt.step()
    res.append([p.detach().float().cpu() for p in list(a.parameters()) + list(b.parameters())])
  return max((x - y).abs().max().item() for x, y in zip(*res))

def some_grads_none():
  m = nn.ModuleDict({"a": nn.Linear(4, 4), "b": nn.Linear(4, 4)}).to(T)
  opt = torch.optim.Adam(m.parameters(), lr=0.1)
  m["a"](torch.ones(1, 4, device=T)).sum().backward(); opt.step()
  return len(opt.state)

def sparse_embedding():
  res = []
  for dev in ("cpu", T):
    torch.manual_seed(0)
    e = nn.Embedding(10, 4, sparse=True).to(dev)
    opt = torch.optim.SparseAdam(e.parameters(), lr=0.1)
    for _ in range(2):
      opt.zero_grad(); e(torch.tensor([1, 2, 2], device=dev)).sum().backward(); opt.step()
    res.append(e.weight.detach().cpu())
  return (res[0] - res[1]).abs().max().item()

def sparse_embedding_sgd():
  res = []
  for dev in ("cpu", T):
    torch.manual_seed(0)
    e = nn.Embedding(10, 4, sparse=True).to(dev)
    opt = torch.optim.SGD(e.parameters(), lr=0.1)
    for _ in range(2):
      opt.zero_grad(); e(torch.tensor([1, 2, 2], device=dev)).sum().backward(); opt.step()
    res.append(e.weight.detach().cpu())
  return (res[0] - res[1]).abs().max().item()

def differentiable_opt():
  w = torch.ones(3, device=T, requires_grad=True)
  lr = torch.tensor(0.1, device=T, requires_grad=True)
  p = w.clone()
  opt = torch.optim.SGD([p], lr=lr, differentiable=True)
  p.grad = torch.ones(3, device=T)
  return "ok"

def param_vector():
  m = nn.Linear(4, 3).to(T)
  v = nn.utils.parameters_to_vector(m.parameters())
  nn.utils.vector_to_parameters(v * 2, m.parameters())
  return v.shape, m.bias.device

def fp16_autocast_train():
  m = nn.Linear(8, 2).to(T); opt = torch.optim.SGD(m.parameters(), lr=0.1)
  for _ in range(3):
    with torch.autocast("tpu", dtype=torch.float16):
      loss = m(torch.ones(4, 8, device=T)).float().pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
  return loss.item(), m.weight.grad.dtype

def clip_bf16():
  r = []
  for dev in ("cpu", T):
    torch.manual_seed(0); m = nn.Linear(16, 16).to(dev).bfloat16()
    m(torch.ones(4, 16, device=dev, dtype=torch.bfloat16)).sum().backward()
    r.append(nn.utils.clip_grad_norm_(m.parameters(), 1.0).float().cpu())
  return r

def ema_lerp_bf16_scalar_tensor():
  ps = [torch.ones(4, device=T, dtype=torch.bfloat16)]
  qs = [torch.zeros(4, device=T, dtype=torch.bfloat16)]
  torch._foreach_lerp_(ps, qs, torch.tensor(0.25, device=T))
  torch._foreach_mul_(ps, torch.tensor(2.0, device=T))
  return ps[0].cpu()

def ddp_bucket_view():
  return "skip"

for f in (mixed_dtype_foreach, some_grads_none, sparse_embedding, sparse_embedding_sgd,
          differentiable_opt, param_vector, fp16_autocast_train, clip_bf16,
          ema_lerp_bf16_scalar_tensor):
  run(f.__name__, f)
