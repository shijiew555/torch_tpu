import torch, torch_tpu, numpy as np, io, traceback
def t(name, f):
    try:
        r = f(); print("OK  ", name, "->", r)
    except Exception as e:
        print("FAIL", name, "->", type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "")
d = "tpu"
x = torch.arange(12, dtype=torch.float32).reshape(3,4)
xt = x.to(d)
t("to tpu cpu roundtrip", lambda: torch.equal(xt.cpu(), x))
t("tensor(device=tpu)", lambda: torch.tensor([1,2,3], device=d).cpu().tolist())
t("tolist", lambda: xt.tolist())
t("item", lambda: xt[1,2].item())
t("transpose cpu", lambda: torch.equal(xt.t().cpu(), x.t()))
t("slice cpu", lambda: torch.equal(xt[:,1:3].cpu(), x[:,1:3]))
t("step slice cpu", lambda: torch.equal(xt[:, ::2].cpu(), x[:, ::2]))
t("non_blocking", lambda: torch.equal(x.to(d, non_blocking=True).cpu(), x))
t("pin_memory", lambda: x.pin_memory().is_pinned())
t("pin non_blocking", lambda: torch.equal(x.pin_memory().to(d, non_blocking=True).cpu(), x))
t("to tpu float16", lambda: x.to(d, torch.float16).cpu().dtype)
t("to tpu bf16 -> cpu", lambda: torch.equal(x.to(d, torch.bfloat16).cpu(), x.bfloat16()))
t("copy_ cpu->tpu dtype", lambda: torch.equal(torch.empty(3,4,device=d,dtype=torch.bfloat16).copy_(x).cpu(), x.bfloat16()))
t("copy_ tpu->cpu dtype", lambda: torch.equal(torch.empty(3,4,dtype=torch.float64).copy_(xt), x.double()))
t("copy_ tpu->cpu noncontig dst", lambda: (lambda c: (c.copy_(xt), torch.equal(c, x))[1])(torch.zeros(4,3).t()))
t("copy_ cpu noncontig -> tpu", lambda: torch.equal(torch.zeros(4,3,device=d).copy_(x.t()).cpu(), x.t()))
t("copy_ into tpu slice", lambda: (lambda z: (z[:,1].copy_(torch.ones(3)), z.cpu().tolist())[1])(torch.zeros(3,4,device=d)))
t("from_numpy", lambda: torch.from_numpy(np.arange(6).reshape(2,3)).to(d).cpu().tolist())
t("from_numpy noncontig", lambda: torch.from_numpy(np.arange(6).reshape(2,3).T).to(d).cpu().tolist())
t("int64 big", lambda: torch.tensor([2**40+1], device=d).item())
t("float64", lambda: torch.tensor([1.0000000001], dtype=torch.float64, device=d).item())
t("bool", lambda: torch.tensor([True, False], device=d).tolist())
t("complex", lambda: torch.tensor([1+2j], device=d).cpu().tolist())
t("uint8", lambda: torch.tensor([255], dtype=torch.uint8, device=d).tolist())
t("device tpu:0", lambda: torch.ones(2, device="tpu:0").device)
t("device obj", lambda: torch.ones(2, device=torch.device("tpu", 0)).device)
t("tpu:1 error", lambda: torch.ones(2, device="tpu:1").device)
def saveload():
    b = io.BytesIO(); torch.save({"a": xt, "v": xt[:, 1]}, b); b.seek(0)
    r = torch.load(b); return r["a"].device, torch.equal(r["a"].cpu(), x), r["v"].cpu().tolist()
t("save/load tpu", saveload)
def saveload_cpu():
    b = io.BytesIO(); torch.save(xt, b); b.seek(0); r = torch.load(b, map_location="cpu"); return r.device, torch.equal(r, x)
t("save/load map cpu", saveload_cpu)
def saveload_tpu():
    b = io.BytesIO(); torch.save(x, b); b.seek(0); r = torch.load(b, map_location="tpu"); return r.device, torch.equal(r.cpu(), x)
t("save cpu load map tpu", saveload_tpu)
def lsd():
    m = torch.nn.Linear(4,2).to(d); sd = torch.nn.Linear(4,2).state_dict()
    m.load_state_dict(sd); return torch.equal(m.weight.cpu(), sd["weight"])
t("load_state_dict cpu->tpu", lsd)
def lsd_assign():
    with torch.device("meta"): m = torch.nn.Linear(4,2)
    sd = {k: v.to(d) for k,v in torch.nn.Linear(4,2).state_dict().items()}
    m.load_state_dict(sd, assign=True); return m.weight.device, m(torch.ones(1,4,device=d)).shape
t("load_state_dict assign", lsd_assign)
def meta_empty():
    with torch.device("meta"): m = torch.nn.Linear(4,2)
    m.to_empty(device=d); m.reset_parameters(); return m.weight.device, m(torch.ones(1,4,device=d)).cpu().shape
t("meta to_empty", meta_empty)
def with_device():
    with torch.device(d): a = torch.randn(3); return a.device
t("with torch.device(tpu)", with_device)
t("set_default_device", lambda: (torch.set_default_device(d), torch.zeros(2).device, torch.set_default_device("cpu"))[1])
