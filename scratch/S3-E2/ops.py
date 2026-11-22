"""torchvision.ops, transforms.v2 and vision/diffusion ops on TPU vs CPU."""
import faulthandler
import traceback

import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401
import torchvision.ops as tvo
import torchvision.transforms.v2 as T

torch.set_float32_matmul_precision("highest")
faulthandler.dump_traceback_later(600, exit=True)
DEV = torch.device("tpu")


def cmp(name, fn, *args, grad=False, tol=1e-4):
  try:
    cargs = [a.clone().requires_grad_(grad and a.is_floating_point()) if isinstance(a, torch.Tensor) else a for a in args]
    targs = [a.detach().to(DEV).requires_grad_(grad and a.is_floating_point()) if isinstance(a, torch.Tensor) else a for a in args]
    oc = fn(*cargs)
    ot = fn(*targs)
    if isinstance(oc, (tuple, list)):
      oc, ot = oc[0], ot[0]
    otc = ot.detach().cpu()
    if oc.shape != otc.shape:
      print(f"MISMATCH {name}: shape cpu={tuple(oc.shape)} tpu={tuple(otc.shape)}", flush=True)
      return
    if oc.is_floating_point():
      d = (otc.float() - oc.detach().float()).abs().max().item() if oc.numel() else 0.0
    else:
      d = (otc != oc).sum().item()
    msg = f"fwd maxdiff={d:.2e}"
    bad = d > tol
    if grad:
      oc.float().pow(2).sum().backward()
      ot.float().pow(2).sum().backward()
      for i, (a, b) in enumerate(zip(cargs, targs)):
        if isinstance(a, torch.Tensor) and a.grad is not None:
          gd = (b.grad.cpu() - a.grad).abs().max().item()
          msg += f" grad{i} maxdiff={gd:.2e}"
          bad = bad or gd > tol * 10
    print(f"{'MISMATCH' if bad else 'OK'} {name}: {msg}", flush=True)
  except Exception as e:  # noqa: BLE001
    tb = traceback.format_exc().strip().splitlines()
    print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:250] if str(e) else ''}", flush=True)
    print("   " + tb[-3].strip()[:200], flush=True)


torch.manual_seed(0)
x = torch.randn(2, 4, 16, 16)
boxes = torch.rand(20, 4) * 30
boxes[:, 2:] += boxes[:, :2]
scores = torch.rand(20)
idxs = torch.randint(0, 3, (20,))
rois = torch.cat([torch.randint(0, 2, (6, 1)).float(), boxes[:6] / 2], 1)

cmp("box_iou", tvo.box_iou, boxes, boxes[:5])
cmp("generalized_box_iou", tvo.generalized_box_iou, boxes, boxes[:5])
cmp("box_convert", lambda b: tvo.box_convert(b, "xyxy", "cxcywh"), boxes)
cmp("clip_boxes_to_image", lambda b: tvo.clip_boxes_to_image(b, (20, 20)), boxes)
cmp("remove_small_boxes", lambda b: tvo.remove_small_boxes(b, 5.0), boxes)
cmp("sigmoid_focal_loss", lambda a, b: tvo.sigmoid_focal_loss(a, b, reduction="mean"), torch.randn(8, 5), (torch.rand(8, 5) > 0.5).float(), grad=True)
cmp("nms", lambda b, s: tvo.nms(b, s, 0.5), boxes, scores)
cmp("batched_nms", lambda b, s, i: tvo.batched_nms(b, s, i, 0.5), boxes, scores, idxs)
cmp("roi_align", lambda a, r: tvo.roi_align(a, r, (3, 3), 1.0, 2, True), x, rois, grad=True)
cmp("roi_pool", lambda a, r: tvo.roi_pool(a, r, (3, 3), 1.0), x, rois, grad=True)
cmp("ps_roi_align", lambda a, r: tvo.ps_roi_align(a, r, (2, 2), 1.0), x, rois)
off = torch.randn(2, 2 * 3 * 3, 14, 14) * 0.5
w = torch.randn(5, 4, 3, 3)
cmp("deform_conv2d", lambda a, o, ww: tvo.deform_conv2d(a, o, ww), x, off, w, grad=True)
cmp("FeaturePyramidNetwork", lambda a: tvo.FeaturePyramidNetwork([4], 8)({"a": a})["a"], x)
cmp("stochastic_depth", lambda a: tvo.stochastic_depth(a, 0.0, "row"), x)

# transforms.v2
img = torch.rand(3, 32, 40)
img8 = (img * 255).to(torch.uint8)
for nm, tr in [
    ("Resize bilinear", T.Resize((20, 24), antialias=False)),
    ("Resize antialias", T.Resize((20, 24), antialias=True)),
    ("Normalize", T.Normalize([0.5, 0.4, 0.3], [0.2, 0.2, 0.2])),
    ("CenterCrop", T.CenterCrop(16)),
    ("HorizontalFlip", T.RandomHorizontalFlip(1.0)),
    ("VerticalFlip", T.RandomVerticalFlip(1.0)),
    ("ColorJitter brightness", T.ColorJitter(brightness=(1.3, 1.3))),
    ("ColorJitter contrast", T.ColorJitter(contrast=(0.7, 0.7))),
    ("ColorJitter saturation", T.ColorJitter(saturation=(0.6, 0.6))),
    ("ColorJitter hue", T.ColorJitter(hue=(0.1, 0.1))),
    ("Grayscale", T.Grayscale()),
    ("GaussianBlur", T.GaussianBlur(5, sigma=1.0)),
    ("Pad reflect", T.Pad(3, padding_mode="reflect")),
    ("RandomRotation fixed", T.RandomRotation((30, 30))),
    ("RandomAffine fixed", T.RandomAffine((10, 10), translate=None, scale=(1.1, 1.1))),
    ("ElasticTransform none", T.functional.equalize),
]:
  cmp("tv2 " + nm, tr, img)
cmp("tv2 Resize uint8", T.Resize((20, 24), antialias=True), img8, tol=1)
cmp("tv2 ToDtype uint8->f32", T.ToDtype(torch.float32, scale=True), img8)
cmp("tv2 equalize uint8", T.functional.equalize, img8, tol=0)
cmp("tv2 posterize uint8", lambda a: T.functional.posterize(a, 3), img8, tol=0)
cmp("tv2 autocontrast", T.functional.autocontrast, img)
cmp("tv2 adjust_sharpness", lambda a: T.functional.adjust_sharpness(a, 2.0), img)

# vision / diffusion ops
v = torch.randn(2, 4, 8, 8)
v3 = torch.randn(1, 4, 4, 6, 6)
for ac in (False, True):
  for mode in ("bilinear", "bicubic"):
    cmp(f"interp {mode} sf2 ac={ac}", lambda a: F.interpolate(a, scale_factor=2, mode=mode, align_corners=ac), v, grad=True)
    cmp(f"interp {mode} sf1.5 ac={ac}", lambda a: F.interpolate(a, scale_factor=1.5, mode=mode, align_corners=ac), v, grad=True)
cmp("interp nearest sf2", lambda a: F.interpolate(a, scale_factor=2.0, mode="nearest"), v, grad=True)
cmp("interp nearest-exact sf0.5", lambda a: F.interpolate(a, scale_factor=0.5, mode="nearest-exact"), v, grad=True)
cmp("interp area", lambda a: F.interpolate(a, size=(3, 5), mode="area"), v, grad=True)
cmp("interp trilinear", lambda a: F.interpolate(a, scale_factor=2, mode="trilinear"), v3, grad=True)
cmp("interp nearest3d", lambda a: F.interpolate(a, scale_factor=2, mode="nearest"), v3, grad=True)
grid = torch.rand(2, 5, 5, 2) * 2.2 - 1.1
for m in ("bilinear", "nearest", "bicubic"):
  for pm in ("zeros", "border", "reflection"):
    cmp(f"grid_sample {m} {pm}", lambda a, g: F.grid_sample(a, g, mode=m, padding_mode=pm, align_corners=False), v, grid, grad=(m != "nearest"))
cmp("affine_grid", lambda t: F.affine_grid(t, (2, 4, 5, 5), align_corners=False), torch.randn(2, 2, 3), grad=True)
cmp("pixel_shuffle", lambda a: F.pixel_shuffle(a, 2), v, grad=True)
cmp("pixel_unshuffle", lambda a: F.pixel_unshuffle(a, 2), v, grad=True)
cmp("group_norm", lambda a, w, b: F.group_norm(a, 2, w, b), v, torch.randn(4), torch.randn(4), grad=True)
cmp("group_norm channels_last", lambda a: F.group_norm(a.to(memory_format=torch.channels_last), 2), v, grad=True)
cmp("conv2d channels_last", lambda a, w: F.conv2d(a.to(memory_format=torch.channels_last), w, padding=1), v, torch.randn(6, 4, 3, 3), grad=True)
cmp("conv3d", lambda a, w: F.conv3d(a, w, padding=1), v3, torch.randn(3, 4, 3, 3, 3), grad=True)
cmp("conv3d stride2", lambda a, w: F.conv3d(a, w, stride=2), v3, torch.randn(3, 4, 2, 2, 2), grad=True)
cmp("conv_transpose3d", lambda a, w: F.conv_transpose3d(a, w, stride=2, padding=1, output_padding=1), v3, torch.randn(4, 3, 3, 3, 3), grad=True)
cmp("conv_transpose2d groups", lambda a, w: F.conv_transpose2d(a, w, stride=2, groups=2), v, torch.randn(4, 3, 3, 3), grad=True)
cmp("max_pool3d", lambda a: F.max_pool3d(a, 2), v3, grad=True)
cmp("max_pool3d pad", lambda a: F.max_pool3d(a, 3, 2, 1), v3, grad=True)
cmp("avg_pool3d", lambda a: F.avg_pool3d(a, 2), v3, grad=True)
cmp("avg_pool3d pad ceil", lambda a: F.avg_pool3d(a, 3, 2, 1, ceil_mode=True, count_include_pad=False), v3, grad=True)
cmp("adaptive_avg_pool3d", lambda a: F.adaptive_avg_pool3d(a, (1, 3, 3)), v3, grad=True)
cmp("adaptive_max_pool2d", lambda a: F.adaptive_max_pool2d(a, (3, 3)), v, grad=True)
cmp("adaptive_max_pool3d", lambda a: F.adaptive_max_pool3d(a, (2, 3, 3)), v3, grad=True)
cmp("max_pool2d ceil", lambda a: F.max_pool2d(a, 3, 2, ceil_mode=True), v, grad=True)
cmp("max_pool2d indices", lambda a: F.max_pool2d(a, 2, return_indices=True)[1], v)
cmp("max_unpool2d", lambda a: F.max_unpool2d(*F.max_pool2d(a, 2, return_indices=True), 2), v, grad=True)
cmp("fractional_max_pool2d", lambda a: F.fractional_max_pool2d(a, 2, output_size=(4, 4), _random_samples=torch.full((2, 4, 2), 0.5).to(a.device)), v, grad=True)
cmp("kthvalue", lambda a: torch.kthvalue(a, 3, dim=-1)[0], v)
cmp("median dim", lambda a: torch.median(a, dim=-1)[0], v)
cmp("instance_norm", lambda a: F.instance_norm(a), v, grad=True)
cmp("local_response_norm", lambda a: F.local_response_norm(a, 3), v, grad=True)
cmp("silu", F.silu, v, grad=True)
cmp("mish", F.mish, v, grad=True)
cmp("hardswish", F.hardswish, v, grad=True)
cmp("pad replicate 3d", lambda a: F.pad(a, (1, 1, 1, 1, 1, 1), mode="replicate"), v3, grad=True)
cmp("pad reflect 3d", lambda a: F.pad(a, (1, 1, 1, 1, 1, 1), mode="reflect"), v3, grad=True)
cmp("pad circular", lambda a: F.pad(a, (1, 1, 1, 1), mode="circular"), v, grad=True)
cmp("channel_shuffle", lambda a: F.channel_shuffle(a, 2), v)
cmp("nn.ChannelShuffle", lambda a: torch.nn.ChannelShuffle(2)(a), v)
cmp("embedding_bag", lambda w: F.embedding_bag(torch.tensor([1, 2, 4, 5]).to(w.device), w, torch.tensor([0, 2]).to(w.device)), torch.randn(10, 3), grad=True)
