import torch, torch_tpu
def t(name, f):
    try:
        r = f(); print("OK  ", name, "->", r, flush=True)
    except Exception as e:
        print("FAIL", name, "->", type(e).__name__, str(e).strip().splitlines()[0][:300] if str(e) else "", flush=True)
d="tpu"
@torch.compile
def drop(x): return torch.nn.functional.dropout(x, 0.5, training=True)
def cdrop():
    x = torch.ones(1000, device=d)
    torch.manual_seed(0); a = drop(x).cpu(); torch.manual_seed(0); b = drop(x).cpu(); c = drop(x).cpu()
    return torch.equal(a,b), torch.equal(b,c), (a==0).float().mean().item()
t("compiled dropout seed repro / advances", cdrop)
@torch.compile
def rn(x): return x + torch.randn_like(x)
def crn():
    x = torch.zeros(8, device=d)
    torch.manual_seed(1); a = rn(x).cpu(); torch.manual_seed(1); b = rn(x).cpu(); c = rn(x).cpu()
    return torch.equal(a,b), torch.equal(b,c)
t("compiled randn_like repro / advances", crn)
def eager_vs_state():
    torch.manual_seed(2); s = torch.accelerator.random.get_rng_state() if hasattr(torch.accelerator, "random") else None
    return type(s)
t("accelerator.random.get_rng_state", eager_vs_state)
@torch.compile
def tocpu(x): return (x * 2).cpu() + 1
t("compiled .cpu()", lambda: tocpu(torch.ones(3, device=d)).tolist())
@torch.compile
def toit(x): return x * x.sum().item()
t("compiled .item()", lambda: toit(torch.ones(3, device=d)).cpu().tolist())
@torch.compile
def todev(x): return x.to(d) * 2
t("compiled cpu->tpu .to", lambda: todev(torch.ones(3)).cpu().tolist())
