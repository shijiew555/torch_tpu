import time, torch, torch_tpu, warnings, logging
import torch._dynamo.utils as du
emb = torch.nn.Linear(64, 64).to("tpu")
def f(x): return emb(x).relu().sum()
cf = torch.compile(f)
for L in range(16, 28):
    x = torch.randn(2, L, 64, device="tpu")
    t = time.time(); cf(x).item()
    print(f"L={L} unique_graphs={du.counters['stats']['unique_graphs']} t={time.time()-t:.2f}s", flush=True)
