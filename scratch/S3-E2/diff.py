"""diffusers tiny UNet2DCondition, AutoencoderKL, DiT, SD3 on TPU vs CPU."""
import copy
import faulthandler
import sys
import time
import traceback

import torch
import torch_tpu  # noqa: F401
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       DiTTransformer2DModel, SD3Transformer2DModel,
                       UNet2DConditionModel)

torch.set_float32_matmul_precision("highest")
faulthandler.dump_traceback_later(1200, exit=True)
DEV = torch.device("tpu")
COMPILE = "--compile" in sys.argv
BF16 = "--bf16" in sys.argv


def unet():
  m = UNet2DConditionModel(
      sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
      block_out_channels=(32, 64), norm_num_groups=8,
      down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
      up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
      cross_attention_dim=32, attention_head_dim=8)
  args = (torch.randn(2, 4, 16, 16), torch.tensor([10, 500]), torch.randn(2, 7, 32))
  return m, args, lambda m, a, b, c: m(a, b, encoder_hidden_states=c).sample


def vae():
  m = AutoencoderKL(
      in_channels=3, out_channels=3, latent_channels=4,
      block_out_channels=(32, 64), norm_num_groups=8, layers_per_block=1,
      down_block_types=("DownEncoderBlock2D",) * 2,
      up_block_types=("UpDecoderBlock2D",) * 2)
  args = (torch.randn(2, 3, 32, 32),)
  return m, args, lambda m, a: m(a, sample_posterior=False).sample


def dit():
  m = DiTTransformer2DModel(
      num_attention_heads=2, attention_head_dim=16, in_channels=4,
      out_channels=8, num_layers=2, sample_size=8, patch_size=2,
      norm_num_groups=8, num_embeds_ada_norm=1000)
  args = (torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), torch.tensor([1, 3]))
  return m, args, lambda m, a, b, c: m(a, timestep=b, class_labels=c).sample


def sd3():
  m = SD3Transformer2DModel(
      sample_size=8, patch_size=2, in_channels=4, num_layers=2,
      attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
      caption_projection_dim=16, pooled_projection_dim=12, out_channels=4,
      pos_embed_max_size=16)
  args = (torch.randn(2, 4, 8, 8), torch.randn(2, 5, 16), torch.randn(2, 12),
          torch.tensor([10.0, 500.0]))
  return m, args, lambda m, a, b, c, d: m(hidden_states=a, encoder_hidden_states=b,
                                          pooled_projections=c, timestep=d).sample


def rel(a, b):
  a = a.detach().float().cpu()
  b = b.detach().float().cpu()
  return ((a - b).norm() / (b.norm() + 1e-12)).item()


names = [a for a in sys.argv[1:] if not a.startswith("--")]
for name in names:
  torch.manual_seed(0)
  try:
    m, args, call = globals()[name]()
    if name == "dit":
      m.eval()  # LabelEmbedding drops class labels at random in train mode.
    mt = copy.deepcopy(m).to(DEV)
    dt = torch.bfloat16 if BF16 else torch.float32
    if BF16:
      m = m.to(dt)
      mt = mt.to(dt)
    ca = [a.to(dt) if a.is_floating_point() else a for a in args]
    ta = [a.to(DEV) for a in ca]
    ft = torch.compile(mt, backend="tpu") if COMPILE else mt
    oc = call(m, *ca)
    oc.float().pow(2).mean().backward()
    t0 = time.time()
    ot = call(ft, *ta)
    ot.float().pow(2).mean().backward()
    ot_c = ot.detach().cpu()
    t1 = time.time()
    gd = 0.0
    gn = 0.0
    for p, q in zip(m.parameters(), mt.parameters()):
      if p.grad is not None:
        gd += (q.grad.cpu().float() - p.grad.float()).norm().item() ** 2
        gn += p.grad.float().norm().item() ** 2
    print(f"OK {name} compile={COMPILE} bf16={BF16} out_rel={rel(ot_c, oc):.2e} "
          f"grad_rel={(gd / gn) ** 0.5:.2e} time={t1 - t0:.1f}s", flush=True)
  except Exception as e:  # noqa: BLE001
    tb = traceback.format_exc().strip().splitlines()
    print(f"FAIL {name} compile={COMPILE} bf16={BF16}: {type(e).__name__}: "
          f"{str(e).splitlines()[0][:300] if str(e) else ''}", flush=True)
    print("   " + "\n   ".join(tb[-6:]), flush=True)

if "--sched" in sys.argv:
  for S in (DDPMScheduler, DDIMScheduler):
    try:
      torch.manual_seed(0)
      m, args, call = unet()
      m.eval()
      mt = copy.deepcopy(m).to(DEV)
      res = []
      for model, dev in ((m, "cpu"), (mt, DEV)):
        s = S(num_train_timesteps=1000)
        s.set_timesteps(5)
        torch.manual_seed(1)
        x = torch.randn(2, 4, 16, 16).to(dev)
        enc = args[2].to(dev)
        with torch.no_grad():
          for t in s.timesteps:
            eps = model(x, t, encoder_hidden_states=enc).sample
            gen = torch.Generator().manual_seed(2)
            if S is DDPMScheduler:
              x = s.step(eps, t, x, generator=gen).prev_sample
            else:
              x = s.step(eps, t, x).prev_sample
        res.append(x.cpu())
      print(f"OK sched {S.__name__} rel={rel(res[1], res[0]):.2e}", flush=True)
    except Exception as e:  # noqa: BLE001
      tb = traceback.format_exc().strip().splitlines()
      print(f"FAIL sched {S.__name__}: {type(e).__name__}: {str(e)[:300]}", flush=True)
      print("   " + "\n   ".join(tb[-6:]), flush=True)
