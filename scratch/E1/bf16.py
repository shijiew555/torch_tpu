import sys, copy, torch, torch_tpu
import transformers as T
sys.path.insert(0, "/root/torch_tpu/.claude/tmp/E1")
from train_eager import build, batch
names=sys.argv[1].split(",")
for name in names:
    # bf16 weights
    try:
        torch.manual_seed(0); m=build(name); mb=copy.deepcopy(m).to(torch.bfloat16)
        mc=copy.deepcopy(mb); mt=mb.to("tpu")
        bc=batch(name,"cpu"); bt=batch(name,"tpu")
        lc=mc(**bc); lt=mt(**bt); lt.loss.backward(); lc.loss.backward()
        gd=max((p1.grad.float()-p2.grad.cpu().float()).abs().max().item() for p1,p2 in zip(mc.parameters(),mt.parameters()) if p1.grad is not None)
        print(f"{name} bf16 loss cpu={lc.loss.item():.4f} tpu={lt.loss.item():.4f} logits dtype={lt.logits.dtype} maxgraddiff={gd:.2e}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} bf16 FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
    # autocast
    try:
        torch.manual_seed(0); m=build(name); mt=copy.deepcopy(m).to("tpu")
        with torch.autocast("cpu",dtype=torch.bfloat16): lc=m(**batch(name,"cpu"))
        with torch.autocast("tpu",dtype=torch.bfloat16): lt=mt(**batch(name,"tpu"))
        lt.loss.backward()
        print(f"{name} autocast loss cpu={lc.loss.item():.4f} tpu={lt.loss.item():.4f} logits dtype cpu={lc.logits.dtype} tpu={lt.logits.dtype} grad dtype={next(mt.parameters()).grad.dtype}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} autocast FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
    try:
        torch.manual_seed(0); m=build(name); mt=copy.deepcopy(m).to("tpu")
        with torch.autocast("tpu"): lt=mt(**batch(name,"tpu"))
        print(f"{name} autocast-default dtype logits={lt.logits.dtype} default={torch.get_autocast_dtype('tpu')}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} autocast-default FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
