import warnings, torch, torch_tpu
warnings.simplefilter("always")
def t(name, f):
    try:
        r = f(); print("OK  ", name, "->", r)
    except Exception as e:
        print("FAIL", name, "->", type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "")
d="tpu"
x = torch.ones(4)
print("--- nb"); x.to(d, non_blocking=True).cpu()
print("--- pin"); p = x.pin_memory()
print("--- is_pinned"); p.is_pinned()
print("--- RNG")
torch.manual_seed(0); a = torch.randn(5, device=d).cpu()
torch.manual_seed(0); b = torch.randn(5, device=d).cpu()
print("manual_seed repro", torch.equal(a,b))
t("tpu.manual_seed", lambda: (torch.tpu.manual_seed(1), torch.randn(2, device=d).cpu())[1])
t("tpu.manual_seed_all", lambda: torch.tpu.manual_seed_all(1))
t("tpu.seed", lambda: torch.tpu.seed())
t("tpu.initial_seed", lambda: torch.tpu.initial_seed())
def rngstate():
    s = torch.tpu.get_rng_state(); a = torch.randn(5, device=d).cpu()
    torch.tpu.set_rng_state(s); b = torch.randn(5, device=d).cpu(); return torch.equal(a,b), s.dtype, s.shape
t("tpu get/set_rng_state", rngstate)
def rngstate_all():
    s = torch.tpu.get_rng_state_all(); torch.tpu.set_rng_state_all(s); return len(s)
t("tpu get/set_rng_state_all", rngstate_all)
def gen():
    g = torch.Generator(device=d); g.manual_seed(3); a = torch.randn(5, device=d, generator=g).cpu()
    g.manual_seed(3); b = torch.randn(5, device=d, generator=g).cpu(); return torch.equal(a,b)
t("Generator(tpu)", gen)
def genstate():
    g = torch.Generator(device=d); g.manual_seed(3); s = g.get_state(); a = torch.rand(5, device=d, generator=g).cpu()
    g.set_state(s); b = torch.rand(5, device=d, generator=g).cpu(); return torch.equal(a,b)
t("Generator get/set_state", genstate)
def gen_indep():
    g = torch.Generator(device=d); g.manual_seed(3); a = torch.randn(5, device=d, generator=g).cpu()
    torch.manual_seed(3); b = torch.randn(5, device=d).cpu(); return torch.equal(a,b)
t("Generator seed == global seed", gen_indep)
def gen_global_unaffected():
    torch.manual_seed(7); a = torch.randn(3, device=d).cpu()
    torch.manual_seed(7); g = torch.Generator(device=d); g.manual_seed(1); torch.randn(3, device=d, generator=g); b = torch.randn(3, device=d).cpu()
    return torch.equal(a,b)
t("explicit gen doesn't advance global", gen_global_unaffected)
def dropout():
    m = torch.nn.Dropout(0.5); y = torch.ones(100, device=d)
    torch.manual_seed(5); a = m(y).cpu(); torch.manual_seed(5); b = m(y).cpu(); return torch.equal(a,b), (a==0).float().mean().item()
t("dropout repro", dropout)
def fork_rng():
    with torch.random.fork_rng(device_type="tpu"):
        torch.randn(3, device=d)
    return True
t("fork_rng device_type tpu", fork_rng)
t("rand_like/randint/normal_/uniform_/bernoulli", lambda: [torch.randint(0,10,(3,),device=d).cpu().tolist(), torch.empty(3,device=d).normal_().cpu().shape, torch.empty(3,device=d).uniform_().cpu().shape, torch.bernoulli(torch.full((3,),0.5,device=d)).cpu().tolist()])
def default_gen():
    g = torch.tpu.default_generators[0] if hasattr(torch.tpu, "default_generators") else None
    return g, g.initial_seed() if g else None
t("default_generators", default_gen)
t("randperm", lambda: torch.randperm(5, device=d).cpu().tolist())
t("multinomial gen", lambda: torch.multinomial(torch.ones(4,device=d), 2, generator=torch.Generator(device=d).manual_seed(0)).cpu().tolist())
print("--- APIs")
for n in ["synchronize","is_available","device_count","current_device","memory_allocated","max_memory_allocated","memory_reserved","empty_cache","memory_stats","reset_peak_memory_stats","get_device_name","get_device_properties","mem_get_info","is_initialized"]:
    f = getattr(torch.tpu, n, None)
    t("torch.tpu."+n, (lambda f=f: f()) if f else (lambda: (_ for _ in ()).throw(AttributeError("missing"))))
for n in ["synchronize","current_accelerator","device_count","is_available","current_device_index","memory_allocated","max_memory_allocated","memory_reserved","empty_cache","memory_stats","reset_peak_memory_stats","current_stream","get_memory_info"]:
    f = getattr(torch.accelerator, n, None)
    t("torch.accelerator."+n, (lambda f=f: f()) if f else (lambda: (_ for _ in ()).throw(AttributeError("missing"))))
t("get_device_module", lambda: torch.get_device_module("tpu"))
t("get_device_module(device)", lambda: torch.get_device_module(torch.device("tpu:0")))
t("tpu.device ctx", lambda: (torch.tpu.device(0).__enter__(), True)[1])
t("set_device", lambda: torch.tpu.set_device(0))
t("accelerator.set_device_index", lambda: torch.accelerator.set_device_index(0))
t("accelerator.device_index ctx", lambda: [None for _ in [torch.accelerator.device_index(0).__enter__()]])
t("Stream", lambda: torch.Stream(device="tpu"))
t("Event", lambda: torch.Event(device="tpu"))
def ev():
    s = torch.Event(device="tpu", enable_timing=True); e = torch.Event(device="tpu", enable_timing=True)
    s.record(); torch.ones(1000,1000,device=d).sum(); e.record(); e.synchronize(); return s.elapsed_time(e)
t("Event timing", ev)
t("tpu.Stream", lambda: torch.tpu.Stream())
t("tpu.current_stream", lambda: torch.tpu.current_stream())
t("tpu.Event", lambda: torch.tpu.Event())
t("amp is_autocast_available", lambda: torch.amp.is_autocast_available("tpu"))
t("tpu.is_bf16_supported", lambda: torch.tpu.is_bf16_supported())
t("torch.backends", lambda: torch.tpu.get_device_capability())
