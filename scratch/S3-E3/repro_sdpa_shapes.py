"""SDPA shape mistakes: one case per process (argv[1]); CPU vs TPU."""
import sys
import torch
import torch.nn.functional as F

CASES = {
  "qk_headdim": ((1, 2, 4, 8), (1, 2, 4, 6), (1, 2, 4, 8)),
  "kv_seqlen": ((1, 2, 4, 8), (1, 2, 5, 8), (1, 2, 6, 8)),
  "batch": ((2, 2, 4, 8), (3, 2, 4, 8), (3, 2, 4, 8)),
  "kv_heads_more": ((1, 2, 4, 8), (1, 4, 4, 8), (1, 4, 4, 8)),
  "kv_heads_mismatch": ((1, 4, 4, 8), (1, 2, 4, 8), (1, 1, 4, 8)),
  "v_headdim_diff_ok": ((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 16)),
  "mask_bad": ((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)),
}
name = sys.argv[1]
qs, ks, vs = CASES[name]
for dev in ["cpu", "tpu"]:
  q, k, v = (torch.randn(s, device=dev) for s in (qs, ks, vs))
  kw = {}
  if name == "mask_bad":
    kw["attn_mask"] = torch.zeros(3, 5, device=dev)
  if name in ("kv_heads_more", "kv_heads_mismatch"):
    kw["enable_gqa"] = True
  try:
    print(dev, "ok", tuple(F.scaled_dot_product_attention(q, k, v, **kw).cpu().shape), flush=True)
  except Exception as e:
    print(dev, type(e).__name__, str(e).replace("\n", " | ")[:300], flush=True)
