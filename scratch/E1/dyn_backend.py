import sys, torch, torch_tpu
import torch._dynamo.utils as du
def g(x): return x * 2
def cat(kv, t): return torch.cat([kv, t], dim=2)
for backend in [None, "tpu"]:
    torch._dynamo.reset(); du.counters.clear()
    cg = torch.compile(g) if backend is None else torch.compile(g, backend=backend)
    for L in [16, 17, 18, 19, 20]:
        cg(torch.ones(2, L, device="tpu")).sum().item()
    a = du.counters["stats"]["unique_graphs"]
    torch._dynamo.reset(); du.counters.clear()
    cc = torch.compile(cat) if backend is None else torch.compile(cat, backend=backend)
    kv = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device="tpu"); t = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device="tpu")
    for _ in range(8): kv = cc(kv, t)
    torch.accelerator.synchronize()
    print(f"backend={backend}: x*2 over 5 lengths -> unique_graphs={a}; doc cat loop 8 steps -> unique_graphs={du.counters['stats']['unique_graphs']} final shape={tuple(kv.shape)}", flush=True)
