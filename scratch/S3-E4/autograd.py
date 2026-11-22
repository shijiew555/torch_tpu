import faulthandler, traceback, warnings
faulthandler.dump_traceback_later(400, exit=True)
import torch, torch_tpu  # noqa
import torch.nn as nn
torch.set_float32_matmul_precision("highest")
D="tpu"
def case(name):
    def deco(f):
        for dev in ("cpu", D):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = f(dev)
                ws = [str(x.message)[:120] for x in w]
                print(f"[{name}] {dev}: {r} warns={ws}")
            except Exception as e:
                print(f"[{name}] {dev}: EXC {type(e).__name__}: {str(e)[:300]}")
        return f
    return deco
def cpu(t): return t.detach().cpu() if isinstance(t, torch.Tensor) else t
def r(t): return [round(v,5) for v in cpu(t).flatten()[:4].tolist()]

@case("register_hook")
def _(d):
    x = torch.arange(4., device=d, requires_grad=True); seen=[]
    y = x*2; y.register_hook(lambda g: (seen.append(g.device.type), g*3)[1])
    x.register_hook(lambda g: g+1)
    y.sum().backward(); return r(x.grad), seen
@case("module_hooks")
def _(d):
    m = nn.Linear(3,2).to(d); log=[]
    m.register_forward_pre_hook(lambda mod, a: (a[0]*2,))
    m.register_full_backward_hook(lambda mod, gi, go: log.append((gi[0].shape, go[0].device.type)))
    m.register_full_backward_pre_hook(lambda mod, go: log.append("pre"))
    torch.manual_seed(0); x = torch.ones(2,3,device=d, requires_grad=True)
    m(x).sum().backward(); return log, r(x.grad)
@case("saved_tensors_hooks")
def _(d):
    packs=[]
    def pack(t): packs.append(t.device.type); return t.cpu()
    def unpack(t): return t.to(d)
    x = torch.linspace(-1,1,6,device=d, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = (x.sin()*x.exp()).sum()
    y.backward(); return r(x.grad), packs
@case("save_on_cpu_pin")
def _(d):
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(8,16), nn.GELU(), nn.Linear(16,4)).to(d)
    x = torch.randn(5,8).to(d).requires_grad_()
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        l = m(x).pow(2).sum()
    l.backward(); return r(x.grad), round(cpu(l).item(),4)
@case("save_on_cpu_nopin")
def _(d):
    x = torch.linspace(-1,1,6,device=d, requires_grad=True)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        y = (x.tanh()*x).sum()
    y.backward(); return r(x.grad)
@case("detect_anomaly_ok")
def _(d):
    x = torch.linspace(-1,1,6,device=d, requires_grad=True)
    with torch.autograd.detect_anomaly():
        (x.sin()*x).sum().backward()
    return r(x.grad)
@case("detect_anomaly_nan")
def _(d):
    x = torch.tensor([0., 1.], device=d, requires_grad=True)
    with torch.autograd.detect_anomaly():
        (x.sqrt()*0 + (x - x).div(x - x + 0)).sum().backward()  # nan grad
    return r(x.grad)
@case("grad_allow_unused")
def _(d):
    a = torch.ones(3,device=d,requires_grad=True); b = torch.ones(3,device=d,requires_grad=True)
    g = torch.autograd.grad((a*2).sum(), [a,b], allow_unused=True, materialize_grads=True)
    g2 = torch.autograd.grad((a*2).sum(), [a,b], allow_unused=True)
    return r(g[0]), r(g[1]), g2[1]
@case("is_grads_batched")
def _(d):
    x = torch.randn(3, device="cpu").to(d).requires_grad_()
    y = x**2
    eye = torch.eye(3, device=d)
    (j,) = torch.autograd.grad(y, x, eye, is_grads_batched=True)
    return r(j.diagonal()), r(2*x)
@case("forward_ad")
def _(d):
    import torch.autograd.forward_ad as fwAD
    p = torch.linspace(0,1,4,device=d); t = torch.ones(4,device=d)
    with fwAD.dual_level():
        dp = fwAD.make_dual(p, t)
        out = (dp.sin()*dp).exp()
        out2 = torch.nn.functional.linear(dp.reshape(2,2), torch.eye(2,device=d)).softmax(-1)
        return r(fwAD.unpack_dual(out).tangent), r(fwAD.unpack_dual(out2).tangent)
@case("inference_mode_later")
def _(d):
    with torch.inference_mode():
        a = torch.ones(3, device=d) * 2
    w = torch.ones(3, device=d, requires_grad=True)
    try:
        (a*w).sum().backward(); res = r(w.grad)
    except Exception as e: res = f"EXC1 {e}"[:120]
    try:
        a.add_(1); res2="inplace ok"
    except Exception as e: res2 = f"EXC2 {str(e)[:80]}"
    b = a.clone(); b.requires_grad_(); return res, res2, b.requires_grad
@case("retain_grad")
def _(d):
    x = torch.ones(3,device=d,requires_grad=True); y = x*3; y.retain_grad(); z=(y**2).sum(); z.backward()
    return r(y.grad), r(x.grad)
@case("requires_grad_view")
def _(d):
    base = torch.arange(6., device=d)
    v = base[1:4]; v.requires_grad_()
    (v*v).sum().backward(); return r(v.grad), v.is_leaf
@case("custom_function")
def _(d):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): ctx.save_for_backward(x); return x.exp()
        @staticmethod
        def backward(ctx, g): (x,) = ctx.saved_tensors; return g*x.exp()*2
    x = torch.linspace(0,1,4,device=d,requires_grad=True); F.apply(x).sum().backward(); return r(x.grad)
@case("double_backward")
def _(d):
    x = torch.linspace(0,1,4,device=d,requires_grad=True)
    (g,) = torch.autograd.grad((x**3).sum(), x, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), x); return r(h)
@case("gradcheck_f64")
def _(d):
    x = torch.randn(3,4,dtype=torch.float64).to(d).requires_grad_()
    return torch.autograd.gradcheck(lambda t: (t.sin()@t.T).tanh(), (x,))
@case("post_accumulate_hook")
def _(d):
    p = torch.ones(3,device=d,requires_grad=True); seen=[]
    p.register_post_accumulate_grad_hook(lambda t: seen.append(r(t.grad)))
    (p*2).sum().backward(); return seen
@case("checkpoint_nonreentrant_nodrop")
def _(d):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(4,4), nn.Tanh()).to(d)
    x = torch.ones(2,4,device=d,requires_grad=True)
    checkpoint(m, x, use_reentrant=False).sum().backward(); return r(x.grad)
