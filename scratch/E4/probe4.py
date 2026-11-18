import torch, torch_tpu, copy, pickle, io, numpy as np, os, tempfile
def t(name, f):
    try:
        r = f(); print("OK  ", name, "->", r, flush=True)
    except Exception as e:
        print("FAIL", name, "->", type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "", flush=True)
d="tpu"
x = torch.arange(12.).reshape(3,4); xt = x.to(d)
t("numpy() on tpu", lambda: xt.numpy())
t("np.asarray", lambda: np.asarray(xt))
t("torch.tensor(tpu)", lambda: torch.tensor(xt).device)
t("as_tensor np device", lambda: torch.as_tensor(np.ones(3), device=d).cpu().tolist())
t("to copy=True", lambda: xt.to(d, copy=True).data_ptr() != xt.data_ptr())
t("to(other)", lambda: x.to(xt).device)
t("channels_last", lambda: torch.equal(torch.arange(24.).reshape(1,2,3,4).to(d).to(memory_format=torch.channels_last).cpu(), torch.arange(24.).reshape(1,2,3,4)))
t("channels_last is_contiguous", lambda: torch.randn(1,2,3,4).to(d).to(memory_format=torch.channels_last).is_contiguous(memory_format=torch.channels_last))
t("0-dim cpu", lambda: xt.sum().cpu())
t("bf16 tolist", lambda: xt.bfloat16().tolist())
t("half item", lambda: xt.half()[0,1].item())
t("repr large", lambda: len(repr(torch.randn(2000, 2000, device=d))))
t("repr int", lambda: repr(torch.tensor([1,2], device=d)))
t("bool if", lambda: bool((xt > 5).any()))
t("deepcopy tensor", lambda: torch.equal(copy.deepcopy(xt).cpu(), x))
def deepcopy_module():
    m = torch.nn.Sequential(torch.nn.Linear(4,4), torch.nn.BatchNorm1d(4)).to(d)
    m2 = copy.deepcopy(m); return m2[0].weight.device, torch.equal(m2[0].weight.cpu(), m[0].weight.cpu()), m2[1].num_batches_tracked.device
t("deepcopy module", deepcopy_module)
def deepcopy_grad():
    m = torch.nn.Linear(4,4).to(d); m(torch.ones(2,4,device=d)).sum().backward()
    m2 = copy.deepcopy(m); return m2.weight.grad is not None and torch.equal(m2.weight.grad.cpu(), m.weight.grad.cpu())
t("deepcopy module w/ grad", deepcopy_grad)
t("pickle", lambda: torch.equal(pickle.loads(pickle.dumps(xt)).cpu(), x))
def save_module():
    b = io.BytesIO(); m = torch.nn.Linear(4,2).to(d); torch.save(m, b); b.seek(0)
    m2 = torch.load(b, weights_only=False); return m2.weight.device
t("save full module", save_module)
def save_view_shared():
    b = io.BytesIO(); torch.save([xt, xt[1]], b); b.seek(0); a, v = torch.load(b)
    a.add_(1); return v.cpu().tolist()  # CPU semantics: view shares storage after load
t("save shared storage view", save_view_shared)
def save_tpu_load_mmap():
    p = tempfile.mktemp(); torch.save({"w": xt}, p); r = torch.load(p, mmap=True, map_location=d); return r["w"].device, torch.equal(r["w"].cpu(), x)
t("load mmap map_location tpu", save_tpu_load_mmap)
def map_loc_dict():
    b = io.BytesIO(); torch.save(xt, b); b.seek(0); return torch.load(b, map_location={"tpu:0": "cpu"}).device
t("map_location dict", map_loc_dict)
def map_loc_dev():
    b = io.BytesIO(); torch.save(xt, b); b.seek(0); return torch.load(b, map_location=torch.device("tpu:0")).device
t("map_location device obj", map_loc_dev)
def bf16_save():
    b = io.BytesIO(); torch.save(xt.bfloat16(), b); b.seek(0); return torch.load(b).dtype
t("save bf16", bf16_save)
def safet():
    from safetensors.torch import save_file, load_file
    p = tempfile.mktemp(); save_file({"w": xt, "b": xt[0].contiguous()}, p)
    r = load_file(p, device="tpu"); return r["w"].device, torch.equal(r["w"].cpu(), x)
t("safetensors save/load device=tpu", safet)
def safet_open():
    from safetensors import safe_open
    from safetensors.torch import save_file
    p = tempfile.mktemp(); save_file({"w": x}, p)
    with safe_open(p, framework="pt", device="tpu:0") as f: return f.get_tensor("w").device
t("safetensors safe_open device tpu:0", safet_open)
def module_to_dtype():
    m = torch.nn.Linear(4,2).to(d, torch.bfloat16); return m.weight.dtype, m.weight.device
t("module.to(d, dtype)", module_to_dtype)
def share_mem():
    return xt.share_memory_().device
t("share_memory_", share_mem)
t("stack", lambda: torch.stack([xt, xt]).shape)
t("cat cpu+tpu error", lambda: torch.cat([x, xt]).device)
t("cpu scalar tensor op", lambda: (xt * torch.tensor(2.)).cpu()[0,1].item())
t("equal across devices", lambda: torch.equal(x, xt))
t("allclose across", lambda: torch.testing.assert_close(xt, x))
t("tpu index with cpu idx", lambda: xt[torch.tensor([0,2])].cpu().tolist())
t("cpu index with tpu idx", lambda: x[torch.tensor([0,2], device=d)].tolist())
t("tpu setitem cpu value", lambda: (lambda z: (z.__setitem__(0, torch.ones(4)), z.cpu()[0].tolist())[1])(torch.zeros(3,4,device=d)))
