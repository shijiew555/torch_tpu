import faulthandler; faulthandler.dump_traceback_later(300, exit=True)
import torch, torch_tpu  # noqa
import torch.multiprocessing as mp
def worker(rank, q):
    x = torch.ones(3, device="tpu") * (rank+1)
    q.put(x.sum().item())
def worker_send(q):
    try:
        q.put(torch.ones(3, device="tpu"))
        print("child put ok")
    except Exception as e:
        print("child put EXC", type(e).__name__, str(e)[:300])
class DS(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return torch.full((4,), float(i))
if __name__ == "__main__":
    import sys
    which = sys.argv[1]
    if which == "spawn":
        ctx = mp.get_context("spawn"); q = ctx.Queue()
        mp.spawn(worker, args=(q,), nprocs=1, join=True); print("spawn result", q.get(timeout=30))
    elif which == "queue_parent":
        ctx = mp.get_context("spawn"); q = ctx.Queue()
        try:
            q.put(torch.ones(3, device="tpu")); print("parent put returned")
            p = ctx.Process(target=print, args=("child alive",)); 
            r = q.get(timeout=30); print("got", r.device, r.cpu())
        except Exception as e: print("parent EXC", type(e).__name__, str(e)[:300])
    elif which == "queue_child":
        ctx = mp.get_context("spawn"); q = ctx.Queue()
        p = ctx.Process(target=worker_send, args=(q,)); p.start()
        try:
            r = q.get(timeout=60); print("parent got", r.device, r.cpu())
        except Exception as e: print("parent get EXC", type(e).__name__, str(e)[:300])
        p.join(60); print("child exit", p.exitcode)
    elif which == "dataloader":
        for kw in [dict(num_workers=2, persistent_workers=True, prefetch_factor=4),
                   dict(num_workers=2, pin_memory=True, pin_memory_device="tpu"),
                   dict(num_workers=2, pin_memory=True, persistent_workers=True),
                   dict(num_workers=0, pin_memory=True, pin_memory_device="tpu")]:
            try:
                dl = torch.utils.data.DataLoader(DS(), batch_size=4, **kw)
                tot = 0.
                for ep in range(2):
                    for b in dl: tot += b.to("tpu", non_blocking=True).sum().item(); pinned = b.is_pinned()
                print("DL", kw, "tot", tot, "pinned", pinned)
            except Exception as e:
                print("DL", kw, "EXC", type(e).__name__, str(e)[:300])
