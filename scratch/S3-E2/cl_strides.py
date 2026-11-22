"""Output strides of common vision ops for channels_last inputs: TPU vs meta.

A stride mismatch between the real TPU kernel and the meta kernel breaks
compiled graphs that later call view() on the result.
"""
import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401

aten = torch.ops.aten
torch.manual_seed(0)
N, C, H, W = 2, 8, 8, 8
cl = torch.channels_last
xc = torch.randn(N, C, H, W).contiguous(memory_format=cl)
gc = torch.randn(N, C, H, W).contiguous(memory_format=cl)
wc = torch.randn(C, C, 3, 3)
rm = torch.zeros(C)
rv = torch.ones(C)
wt = torch.randn(C)


def cases(x, g, w, r_m, r_v, wn):
  yield "relu", torch.relu(x)
  yield "silu_backward", aten.silu_backward(g, x)
  yield "gelu_backward", aten.gelu_backward(g, x)
  yield "threshold_backward", aten.threshold_backward(g, x, 0.0)
  yield "conv2d", F.conv2d(x, w, padding=1)
  cb = aten.convolution_backward(g, x, w, [C], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
  yield "convolution_backward[0]", cb[0]
  y, sm, sv = aten.native_batch_norm(x, wn, wn, r_m.clone(), r_v.clone(), True, 0.1, 1e-5)
  yield "native_batch_norm", y
  yield "native_batch_norm_backward[0]", aten.native_batch_norm_backward(g, x, wn, r_m, r_v, sm, sv, True, 1e-5, [True, True, True])[0]
  yg, gm, gr = aten.native_group_norm(x, wn, wn, N, C, H * W, 2, 1e-5)
  yield "native_group_norm", yg
  yield "native_group_norm_backward[0]", aten.native_group_norm_backward(g, x, gm, gr, wn, N, C, H * W, 2, [True, True, True])[0]
  p, idx = aten.max_pool2d_with_indices(x, [2, 2], [2, 2])
  yield "max_pool2d_with_indices", p
  gp = torch.ones_like(p)
  yield "max_pool2d_with_indices_backward", aten.max_pool2d_with_indices_backward(gp, x, [2, 2], [2, 2], [0, 0], [1, 1], False, idx)
  yield "avg_pool2d", F.avg_pool2d(x, 2)
  yield "avg_pool2d_backward", aten.avg_pool2d_backward(gp, x, [2, 2], [2, 2], [0, 0], False, True, None)
  yield "adaptive_avg_pool2d", aten._adaptive_avg_pool2d(x, [1, 1])
  yield "adaptive_avg_pool2d_backward", aten._adaptive_avg_pool2d_backward(torch.ones(N, C, 1, 1, device=x.device).contiguous(memory_format=cl), x)
  up = aten.upsample_bilinear2d(x, [16, 16], False)
  yield "upsample_bilinear2d", up
  yield "upsample_bilinear2d_backward", aten.upsample_bilinear2d_backward(torch.ones_like(up), [16, 16], [N, C, H, W], False)
  yield "upsample_nearest2d", aten.upsample_nearest2d(x, [16, 16])
  yield "pixel_shuffle", F.pixel_shuffle(x, 2)
  yield "native_dropout", aten.native_dropout(x, 0.0, True)[0]
  yield "hardswish_backward", aten.hardswish_backward(g, x)
  yield "sigmoid_backward", aten.sigmoid_backward(g, x)
  yield "instance_norm", F.instance_norm(x)


def collect(dev):
  args = [t.to(dev) for t in (xc, gc, wc, rm, rv, wt)]
  out = {}
  it = cases(*args)
  while True:
    try:
      name, t = next(it)
      out[name] = t.stride()
    except StopIteration:
      break
    except Exception as e:  # noqa: BLE001
      out[f"error {len(out)}"] = f"{type(e).__name__}: {str(e)[:100]}"
      break
  return out


meta = collect("meta")
cpu = collect("cpu")
tpu = collect("tpu")
for k in cpu:
  flag = "MISMATCH" if tpu.get(k) != meta.get(k) else "ok"
  print(f"{flag:8s} {k:36s} cpu={cpu[k]} meta={meta.get(k)} tpu={tpu.get(k)}")
for k in tpu:
  if k not in cpu:
    print("tpu-only", k, tpu[k])
