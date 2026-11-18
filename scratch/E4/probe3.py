import torch, torch_tpu
from torch.utils.data import DataLoader, TensorDataset
def t(name, f):
    try:
        r = f(); print("OK  ", name, "->", r, flush=True)
    except Exception as e:
        print("FAIL", name, "->", type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "", flush=True)
d="tpu"
t("Stream('tpu')", lambda: torch.Stream("tpu"))
t("Stream('tpu:0')", lambda: torch.Stream("tpu:0"))
t("Stream(device=current_accelerator())", lambda: torch.Stream(device=torch.accelerator.current_accelerator()))
t("accelerator.current_stream('tpu')", lambda: torch.accelerator.current_stream(torch.device("tpu")))
t("accelerator.synchronize('tpu')", lambda: torch.accelerator.synchronize(torch.device("tpu")))
def stream_ctx():
    s = torch.Stream("tpu:0")
    with s:
        y = torch.ones(3, device=d) * 2
    s.synchronize(); return y.cpu().tolist()
t("stream ctx", stream_ctx)
def d2h_nb():
    x = torch.randn(1000, 1000, device=d); ref = x.cpu()
    y = x.to("cpu", non_blocking=True); torch.accelerator.synchronize(); return torch.equal(y, ref), y.is_pinned()
t("D2H non_blocking + sync", d2h_nb)
def d2h_nb_event():
    x = torch.randn(1000, 1000, device=d) + 1; ref = x.cpu()
    y = x.to("cpu", non_blocking=True); e = torch.Event(device="tpu"); e.record(); e.synchronize(); return torch.equal(y, ref)
t("D2H non_blocking + event", d2h_nb_event)
def h2d_nb_mutate():
    x = torch.ones(1000, 1000)
    y = x.to(d, non_blocking=True); x.fill_(5); torch.accelerator.synchronize(); return y.cpu()[0,0].item()
t("H2D non_blocking then mutate src (CUDA unpinned => 1.0)", h2d_nb_mutate)
def h2d_nb_pinned_mutate():
    x = torch.ones(1000, 1000).pin_memory()
    y = x.to(d, non_blocking=True); torch.accelerator.synchronize(); x.fill_(5); return y.cpu()[0,0].item()
t("H2D pinned nb, sync, mutate", h2d_nb_pinned_mutate)
def copy_nb():
    z = torch.empty(1000, device=d); z.copy_(torch.arange(1000.), non_blocking=True); return torch.equal(z.cpu(), torch.arange(1000.))
t("copy_ non_blocking", copy_nb)
def dl(nw, pin):
    ds = TensorDataset(torch.arange(64.).reshape(32,2), torch.arange(32))
    loader = DataLoader(ds, batch_size=8, num_workers=nw, pin_memory=pin, shuffle=False)
    tot = 0.
    for a, b in loader:
        p = a.is_pinned()
        a = a.to(d, non_blocking=True); b = b.to(d, non_blocking=True)
        tot += (a.sum() + b.sum()).item()
    return tot, p
t("DataLoader nw=0 pin", lambda: dl(0, True))
t("DataLoader nw=2 pin=False", lambda: dl(2, False))
t("DataLoader nw=2 pin=True", lambda: dl(2, True))
def dl_dev():
    ds = TensorDataset(torch.arange(64.).reshape(32,2))
    loader = DataLoader(ds, batch_size=8, num_workers=2, pin_memory=True, pin_memory_device="tpu")
    return sum(a[0].to(d).sum().item() for a in loader)
t("DataLoader pin_memory_device=tpu", dl_dev)
