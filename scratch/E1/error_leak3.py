import copy, torch, torch_tpu
from train_eager import build, batch
torch.manual_seed(0)
mt = build("bert").to("tpu")
with torch.autocast("tpu"):
    out = mt(**batch("bert", "tpu"))  # layer_norm lowering fails (#5678); result not read
print("bert forward returned (lazy)", flush=True)
try:
    m2 = build("t5").to(torch.bfloat16).to("tpu")
    print("unrelated t5 .to('tpu') OK", flush=True)
except Exception as e:
    print("unrelated t5 .to('tpu') FAILED:", type(e).__name__, str(e)[:200], flush=True)
try:
    print("small op:", (torch.ones(3, device="tpu") + 1).cpu().tolist(), flush=True)
except Exception as e:
    print("small op FAILED:", type(e).__name__, str(e)[:200], flush=True)
