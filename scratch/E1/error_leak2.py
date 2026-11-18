import torch, torch_tpu
ln = torch.nn.LayerNorm(8).to("tpu")
with torch.autocast("tpu", dtype=torch.bfloat16):
    bad = ln(torch.randn(4, 8, device="tpu"))  # known failing lowering (#5678), never read
pass
x = torch.arange(6.0).reshape(2, 3)
try:
    y = x.to("tpu")
    print("unrelated .to('tpu') + .cpu():", (y * 2).cpu().tolist())
except Exception as e:
    print("unrelated op FAILED:", type(e).__name__, str(e)[:200])
try:
    print("second unrelated op:", (torch.ones(3, device="tpu") + 1).cpu().tolist())
except Exception as e:
    print("second unrelated op FAILED:", type(e).__name__, str(e)[:200])
