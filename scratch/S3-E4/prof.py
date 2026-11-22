import faulthandler, os, json, glob, warnings
faulthandler.dump_traceback_later(400, exit=True)
import torch, torch_tpu  # noqa
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler
d="tpu"
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(256,512), nn.ReLU(), nn.Linear(512,10)).to(d)
opt = torch.optim.SGD(m.parameters(), lr=0.1)
x = torch.randn(64,256).to(d); y = torch.randint(0,10,(64,)).to(d)
def step():
    with record_function("my_step"):
        opt.zero_grad(); l = nn.functional.cross_entropy(m(x), y); l.backward(); opt.step()
    return l
step(); torch.accelerator.synchronize()
print("PrivateUse1 activity:", ProfilerActivity.PrivateUse1)
for acts in ([ProfilerActivity.CPU], [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with profile(activities=acts, record_shapes=True, profile_memory=True) as p:
                step().item()
        print("ACTS", acts, "warns", [str(i.message)[:150] for i in w])
        ka = p.key_averages()
        names = [e.key for e in ka]
        print(" n events", len(names), "has my_step", "my_step" in names)
        dev = [e for e in ka if getattr(e, "device_time_total", 0) > 0]
        print(" events with device time:", len(dev), [e.key for e in dev[:5]])
        print(" device types:", set(str(e.device_type) for e in p.events()))
        print(ka.table(sort_by="self_cpu_time_total", row_limit=8)[:2500])
        p.export_chrome_trace(f"/tmp/s3e4_trace_{len(acts)}.json")
        tr = json.load(open(f"/tmp/s3e4_trace_{len(acts)}.json"))
        cats = {}
        for e in tr["traceEvents"]:
            cats[e.get("cat")] = cats.get(e.get("cat"),0)+1
        print(" chrome trace cats", cats)
    except Exception as e:
        import traceback; traceback.print_exc()
# schedule + tensorboard
try:
    out = "/tmp/s3e4_tb"; os.system(f"rm -rf {out}")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                 schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
                 on_trace_ready=tensorboard_trace_handler(out)) as p:
        for _ in range(5):
            step().item(); p.step()
    print("tb files", glob.glob(out+"/*"))
except Exception:
    import traceback; traceback.print_exc()
# memory stats
acc = torch.accelerator
try:
    acc.reset_peak_memory_stats()
    a0 = acc.memory_allocated(); 
    big = torch.empty(1024,1024,64, device=d)  # 256MB
    a1 = acc.memory_allocated(); pk = acc.max_memory_allocated()
    del big; a2 = acc.memory_allocated(); pk2 = acc.max_memory_allocated()
    acc.reset_peak_memory_stats(); pk3 = acc.max_memory_allocated()
    print("mem alloc before/after/del:", a0, a1, a2, "peak", pk, pk2, "peak after reset", pk3)
    print("memory_reserved", acc.memory_reserved(), "stats keys", list(acc.memory_stats().keys())[:10])
    print("torch.tpu.memory_allocated", "skip")
    print("mem_get_info", torch.accelerator.get_memory_info() if hasattr(torch.accelerator,"get_memory_info") else "n/a")
except Exception:
    import traceback; traceback.print_exc()
