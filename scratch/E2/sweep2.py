import sys
import traceback

import torch
import torch_tpu  # noqa: F401

print("torch_tpu file:", torch_tpu.__file__)
D = "tpu"
BACKEND = "tpu"
results = []
ONLY = sys.argv[1:]


def close(a, b, tol=1e-2):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().float()
        b = b.detach().cpu().float()
        return a.shape == b.shape and torch.allclose(a, b, atol=tol, rtol=tol)
    return a == b


def case(name):
    def deco(fn):
        if ONLY and name not in ONLY:
            return fn
        torch._dynamo.reset()
        try:
            r = fn()
            results.append((name, "OK" if r in (None, True) else f"MISMATCH {r}"))
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()
            results.append((name, f"ERR {type(e).__name__}: {str(e)[:300]}"))
            print("=====", name, "\n", "\n".join(tb[-25:]))
        print(name, results[-1][1], flush=True)
        return fn

    return deco


def C(fn, **kw):
    return torch.compile(fn, backend=BACKEND, **kw)


@case("mutate_then_return_view_of_input")
def _():
    def f(x):
        x.mul_(2)
        return x.view(-1)

    x = torch.randn(4, 4)
    xt = x.to(D)
    o = C(f)(xt)
    o.add_(1)  # should change xt too
    return close(xt, x * 2 + 1) or (xt.cpu(), x * 2 + 1)


@case("return_view_then_mutate_output")
def _():
    def f(x):
        return x[0]

    x = torch.randn(4, 4)
    xt = x.to(D)
    o = C(f)(xt * 1)
    xt2 = xt.clone()
    o2 = C(f)(xt2)
    o2.fill_(7)
    return close(xt2[0], torch.full((4,), 7.0)) or xt2.cpu()


@case("noncontig_input_inplace")
def _():
    def f(x):
        x.add_(1)

    x = torch.randn(4, 6)
    xt = x.to(D)
    C(f)(xt.t())
    return close(xt, x + 1) or xt.cpu()


@case("slice_input_inplace")
def _():
    def f(x):
        x.mul_(10)

    x = torch.randn(6, 4)
    xt = x.to(D)
    C(f)(xt[2:4])
    x[2:4].mul_(10)
    return close(xt, x) or xt.cpu()


@case("ema_param_update_no_grad")
def _():
    a = torch.nn.Linear(4, 4)
    b = torch.nn.Linear(4, 4)
    at = torch.nn.Linear(4, 4).to(D)
    bt = torch.nn.Linear(4, 4).to(D)
    at.load_state_dict(a.state_dict())
    bt.load_state_dict(b.state_dict())

    @torch.no_grad()
    def ema(m1, m2):
        for p, q in zip(m1.parameters(), m2.parameters()):
            p.mul_(0.9).add_(q, alpha=0.1)

    ce = C(ema)
    for _ in range(3):
        ce(at, bt)
        ema(a, b)
    return close(at.weight, a.weight, 1e-4) or (at.weight.cpu() - a.weight).abs().max()


@case("dropout_train_compiled_stats")
def _():
    def f(x):
        return torch.nn.functional.dropout(x, 0.5, training=True)

    x = torch.ones(1000, 100).to(D)
    o = C(f)(x).cpu()
    frac = (o == 0).float().mean().item()
    ok = 0.45 < frac < 0.55 and torch.allclose(o[o != 0], torch.tensor(2.0))
    # two calls should differ
    o2 = C(f)(x).cpu()
    return (ok and not torch.equal(o, o2)) or (frac, torch.equal(o, o2))


@case("rand_seed_reproducible")
def _():
    def f(x):
        return x + torch.rand_like(x)

    cf = C(f)
    x = torch.zeros(8).to(D)
    torch.manual_seed(1)
    a = cf(x).cpu()
    torch.manual_seed(1)
    b = cf(x).cpu()
    return torch.equal(a, b) or (a, b)


@case("torch_cond")
def _():
    def f(x):
        return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))

    x = torch.randn(4)
    out = [close(C(f)(v.to(D)), f(v)) for v in (x.abs(), -x.abs())]
    return all(out) or out


@case("nonzero_graph_break")
def _():
    def f(x):
        idx = torch.nonzero(x > 0)
        return idx.sum() + x.sum()

    x = torch.randn(10)
    return close(C(f)(x.to(D)), f(x))


@case("bool_mask_index")
def _():
    def f(x):
        return x[x > 0].sum() * 2

    x = torch.randn(10)
    return close(C(f)(x.to(D)), f(x))


@case("dtype_recompile")
def _():
    def f(x):
        return x * 2 + 1

    cf = C(f)
    out = []
    for dt in (torch.float32, torch.bfloat16, torch.int32, torch.int64):
        x = torch.arange(6).to(dt)
        out.append(close(cf(x.to(D)), f(x)))
    return all(out) or out


@case("mark_dynamic_backward")
def _():
    lin = torch.nn.Linear(5, 3)
    lt = torch.nn.Linear(5, 3).to(D)
    lt.load_state_dict(lin.state_dict())
    cm = C(lt)
    out = []
    for n in (4, 9, 17):
        x = torch.randn(n, 5)
        xt = x.to(D)
        torch._dynamo.mark_dynamic(xt, 0)
        lt.zero_grad()
        lin.zero_grad()
        cm(xt).pow(2).mean().backward()
        lin(x).pow(2).mean().backward()
        out.append(close(lt.weight.grad, lin.weight.grad))
    return all(out) or out


@case("mark_dynamic_view_reshape")
def _():
    def f(x):
        b = x.shape[0]
        return x.reshape(b, -1, 2).transpose(1, 2).sum(-1)

    cf = C(f)
    out = []
    for n in (3, 6, 10):
        x = torch.randn(n, 8)
        xt = x.to(D)
        torch._dynamo.mark_dynamic(xt, 0)
        out.append(close(cf(xt), f(x)))
    return all(out) or out


@case("mark_dynamic_inplace_input")
def _():
    def f(x, y):
        x.add_(y)
        return x.sum(0)

    cf = C(f)
    out = []
    for n in (3, 6, 10):
        x = torch.randn(n, 4)
        y = torch.randn(n, 4)
        xt, yt = x.to(D), y.to(D)
        torch._dynamo.mark_dynamic(xt, 0)
        torch._dynamo.mark_dynamic(yt, 0)
        o = cf(xt, yt)
        oc = f(x, y)
        out.append(close(o, oc) and close(xt, x))
    return all(out) or out


@case("mark_dynamic_return_slice_alias")
def _():
    def f(x):
        n = x.shape[0]
        return x[: n // 2]

    cf = C(f)
    out = []
    for n in (4, 8, 12):
        x = torch.randn(n, 3)
        xt = x.to(D)
        torch._dynamo.mark_dynamic(xt, 0)
        o = cf(xt)
        out.append(close(o, f(x)))
    return all(out) or out


@case("python_scalar_args_recompile")
def _():
    def f(x, n: int, s: float):
        return x.repeat(n, 1) * s

    cf = C(f)
    out = []
    for n, s in ((1, 0.5), (2, 1.5), (3, 2.5)):
        x = torch.randn(2, 3)
        out.append(close(cf(x.to(D), n, s), f(x, n, s)))
    return all(out) or out


@case("kwargs_and_none_inputs")
def _():
    def f(x, bias=None, *, scale=1.0):
        y = x * scale
        if bias is not None:
            y = y + bias
        return y

    cf = C(f)
    x = torch.randn(3)
    b = torch.randn(3)
    return close(cf(x.to(D), None, scale=2.0), f(x, None, scale=2.0)) and close(
        cf(x.to(D), b.to(D), scale=3.0), f(x, b, scale=3.0)
    )


@case("compile_module_method_and_hooks")
def _():
    m = torch.nn.Linear(4, 4)
    mt = torch.nn.Linear(4, 4).to(D)
    mt.load_state_dict(m.state_dict())
    seen = []
    mt.register_forward_hook(lambda mod, i, o: seen.append(o.shape))
    m.register_forward_hook(lambda mod, i, o: None)
    mt.compile(backend=BACKEND)
    x = torch.randn(2, 4)
    return close(mt(x.to(D)), m(x)) and len(seen) == 1 or seen


@case("grad_accumulation_two_backwards")
def _():
    m = torch.nn.Linear(4, 4)
    mt = torch.nn.Linear(4, 4).to(D)
    mt.load_state_dict(m.state_dict())
    cm = C(mt)
    for _ in range(2):
        x = torch.randn(3, 4)
        cm(x.to(D)).sum().backward()
        m(x).sum().backward()
    return close(mt.weight.grad, m.weight.grad)


@case("complex_ops")
def _():
    def f(x):
        z = torch.view_as_complex(x)
        return torch.view_as_real(z * z)

    x = torch.randn(4, 2)
    return close(C(f)(x.to(D)), f(x))


@case("sdpa_compiled_backward")
def _():
    q = torch.randn(2, 4, 16, 8)
    def f(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

    qt = q.to(D).requires_grad_()
    qc = q.clone().requires_grad_()
    o = C(f)(qt, qt, qt)
    o.sum().backward()
    f(qc, qc, qc).sum().backward()
    return close(o, f(q, q, q)) and close(qt.grad, qc.grad)


@case("autocast_bf16_inside_compile")
def _():
    lin = torch.nn.Linear(8, 8)
    lt = torch.nn.Linear(8, 8).to(D)
    lt.load_state_dict(lin.state_dict())

    def f(m, x):
        with torch.autocast("tpu", dtype=torch.bfloat16):
            return m(x)

    x = torch.randn(4, 8)
    o = C(f)(lt, x.to(D))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        oc = lin(x)
    return (o.dtype == torch.bfloat16 and close(o, oc, 5e-2)) or (o.dtype,)


print("\nSUMMARY")
for n, r in results:
    print(f"{n:35s} {r[:200]}")
