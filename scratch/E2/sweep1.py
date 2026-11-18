import sys
import traceback

import torch
import torch_tpu  # noqa: F401

print("torch_tpu file:", torch_tpu.__file__)
D = "tpu"
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "tpu"
results = []


def close(a, b, tol=1e-2):
    if isinstance(a, (tuple, list)):
        return all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().float()
        b = b.detach().cpu().float()
        return a.shape == b.shape and torch.allclose(a, b, atol=tol, rtol=tol)
    return a == b


def case(name):
    def deco(fn):
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


@case("print_graph_break")
def _():
    def f(x):
        y = x * 2
        print("hi")
        return y + 1

    x = torch.randn(4, 4)
    return close(C(f)(x.to(D)), f(x))


@case("item_graph_break")
def _():
    def f(x):
        s = x.sum().item()
        return x * s

    x = torch.randn(4, 4)
    return close(C(f)(x.to(D)), f(x))


@case("data_dependent_if")
def _():
    def f(x):
        if x.sum() > 0:
            return x + 1
        return x - 1

    x = torch.randn(4, 4)
    out = [close(C(f)(v.to(D)), f(v)) for v in (x, -x.abs(), x.abs())]
    return all(out) or out


@case("inplace_input_mutation")
def _():
    def f(x):
        x.add_(1)
        return x * 2

    x = torch.randn(4, 4)
    xt = x.to(D)
    xc = x.clone()
    o = C(f)(xt)
    oc = f(xc)
    return (close(o, oc) and close(xt, xc)) or (o.cpu(), xt.cpu(), xc)


@case("inplace_input_view_mutation")
def _():
    def f(x):
        x[:, 0].mul_(3)
        return x.sum()

    x = torch.randn(4, 4)
    xt = x.to(D)
    xc = x.clone()
    o = C(f)(xt)
    oc = f(xc)
    return (close(o, oc) and close(xt, xc)) or (o.cpu(), xt.cpu(), xc)


@case("buffer_mutation_module")
def _():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cnt", torch.zeros(3))
            self.lin = torch.nn.Linear(3, 3)

        def forward(self, x):
            self.cnt += 1
            return self.lin(x) + self.cnt

    torch.manual_seed(0)
    m = M()
    mt = M().to(D)
    mt.load_state_dict(m.state_dict())
    cm = C(mt)
    x = torch.randn(2, 3)
    for _ in range(3):
        o = cm(x.to(D))
        oc = m(x)
    return (close(o, oc) and close(mt.cnt, m.cnt)) or (mt.cnt.cpu(), m.cnt)


@case("no_grad")
def _():
    def f(x, w):
        with torch.no_grad():
            y = x @ w
        return y * 2 + x.sum()

    x = torch.randn(4, 4)
    w = torch.randn(4, 4, requires_grad=True)
    o = C(f)(x.to(D), w.detach().to(D).requires_grad_())
    return close(o, f(x, w)) and not o.requires_grad


@case("inference_mode_outside")
def _():
    def f(x):
        return torch.sin(x) * 2

    x = torch.randn(4, 4)
    with torch.inference_mode():
        o = C(f)(x.to(D))
    return close(o, f(x))


@case("inference_mode_inside")
def _():
    def f(x):
        with torch.inference_mode():
            y = torch.cos(x)
        return y + 1

    x = torch.randn(4, 4)
    return close(C(f)(x.to(D)), f(x))


@case("requires_grad_change")
def _():
    def f(x):
        return (x * x).sum()

    x = torch.randn(4, 4)
    cf = C(f)
    a = cf(x.to(D))
    xt = x.to(D).requires_grad_()
    b = cf(xt)
    b.backward()
    return close(a, f(x)) and close(xt.grad, 2 * x)


@case("return_view_of_input")
def _():
    def f(x):
        return x[1:3], x.t(), x.view(-1)

    x = torch.randn(4, 4)
    xt = x.to(D)
    o = C(f)(xt)
    # Mutate input; views should reflect it
    xt.add_(1)
    oc = f(x + 1)
    return close(o, oc) or [a.cpu() for a in o]


@case("return_input_itself")
def _():
    def f(x, y):
        return x, y + 1

    x = torch.randn(4)
    xt = x.to(D)
    o = C(f)(xt, xt)
    return (o[0] is xt or o[0].data_ptr() == xt.data_ptr()) or "not same"


@case("outputs_alias_each_other")
def _():
    def f(x):
        y = x * 2
        return y, y.view(2, 8), y[0]

    x = torch.randn(4, 4)
    o = C(f)(x.to(D))
    o[0].add_(1)
    ok = close(o[1], (x * 2 + 1).view(2, 8))
    return ok or (o[1].cpu(), (x * 2 + 1).view(2, 8))


@case("dynamic_true")
def _():
    def f(x):
        return torch.nn.functional.softmax(x, -1).sum(0) + x.shape[0]

    cf = C(f, dynamic=True)
    out = []
    for n in (3, 5, 7, 11):
        x = torch.randn(n, 6)
        out.append(close(cf(x.to(D)), f(x)))
    return all(out) or out


@case("mark_dynamic")
def _():
    def f(x):
        return (x.cumsum(0) * x.size(0)).mean(1)

    cf = C(f)
    out = []
    for n in (4, 8, 13):
        x = torch.randn(n, 5)
        xt = x.to(D)
        torch._dynamo.mark_dynamic(xt, 0)
        out.append(close(cf(xt), f(x)))
    return all(out) or out


@case("higher_order_grad")
def _():
    def f(x):
        return (x**3).sum()

    x = torch.randn(5)
    xt = x.to(D).requires_grad_()
    y = C(f)(xt)
    (g,) = torch.autograd.grad(y, xt, create_graph=True)
    (g2,) = torch.autograd.grad(g.sum(), xt)
    return (close(g, 3 * x * x) and close(g2, 6 * x)) or (g2.cpu(), 6 * x)


@case("compiled_grad_inside")
def _():
    def f(x):
        y = (x**3).sum()
        (g,) = torch.autograd.grad(y, x, create_graph=True)
        return g.sum()

    x = torch.randn(5)
    xt = x.to(D).requires_grad_()
    o = C(f)(xt)
    return close(o, (3 * x * x).sum())


@case("retain_graph")
def _():
    lin = torch.nn.Linear(4, 4)
    lt = torch.nn.Linear(4, 4).to(D)
    lt.load_state_dict(lin.state_dict())
    cm = C(lt)
    x = torch.randn(3, 4)
    o = cm(x.to(D)).sum()
    o.backward(retain_graph=True)
    o.backward()
    oc = lin(x).sum()
    oc.backward(retain_graph=True)
    oc.backward()
    return close(lt.weight.grad, lin.weight.grad)


@case("detach_inside")
def _():
    def f(x):
        return (x * x.detach()).sum()

    x = torch.randn(5)
    xt = x.to(D).requires_grad_()
    C(f)(xt).backward()
    return close(xt.grad, x)


class Sq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * 2 * x


@case("custom_autograd_function")
def _():
    def f(x):
        return Sq.apply(x).sum()

    x = torch.randn(5)
    xt = x.to(D).requires_grad_()
    C(f, fullgraph=True)(xt).backward()
    return close(xt.grad, 2 * x)


@case("checkpoint_inside")
def _():
    import torch.utils.checkpoint as cp

    lin = torch.nn.Linear(8, 8)
    lt = torch.nn.Linear(8, 8).to(D)
    lt.load_state_dict(lin.state_dict())

    def body(m, x):
        return torch.relu(m(x)).sin()

    def f(m, x):
        return cp.checkpoint(body, m, x, use_reentrant=False).sum()

    x = torch.randn(4, 8)
    C(f)(lt, x.to(D)).backward()
    f(lin, x).backward()
    return close(lt.weight.grad, lin.weight.grad)


@case("fullgraph_item_error")
def _():
    def f(x):
        return x * x.sum().item()

    try:
        C(f, fullgraph=True)(torch.randn(3).to(D))
    except Exception as e:
        print("fullgraph msg:", type(e).__name__, str(e)[:500])
        return True
    return "no error"


@case("recompile_limit")
def _():
    def f(x, k):
        return x * k

    cf = C(f)
    x = torch.randn(3).to(D)
    for k in range(12):
        cf(x, float(k) + 0.5)
    return True


@case("module_train_eval_switch")
def _():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
    mt = m.to(D)
    cm = C(mt)
    x = torch.randn(3, 4).to(D)
    cm.train()
    cm(x)
    cm.eval()
    a = cm(x)
    b = mt(x)
    return close(a, b)


@case("int_and_bool_outputs")
def _():
    def f(x):
        return x.argmax(-1), x > 0, x.long() + 1

    x = torch.randn(4, 4) * 3
    return close(C(f)(x.to(D)), f(x))


@case("scalar_cpu_tensor_input")
def _():
    def f(x, s):
        return x * s

    x = torch.randn(4)
    s = torch.tensor(3.0)
    return close(C(f)(x.to(D), s), f(x, s))


@case("set_grad_enabled_false_outside")
def _():
    def f(x):
        return x.exp()

    x = torch.randn(4)
    xt = x.to(D).requires_grad_()
    with torch.no_grad():
        o = C(f)(xt)
    return close(o, x.exp()) and not o.requires_grad


@case("optimizer_step_compiled")
def _():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 2)
    mt = torch.nn.Linear(4, 2).to(D)
    mt.load_state_dict(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    optt = torch.optim.AdamW(mt.parameters(), lr=0.1)

    @torch.compile(backend=BACKEND)
    def step():
        optt.step()

    x = torch.randn(3, 4)
    for _ in range(3):
        m(x).sum().backward()
        mt(x.to(D)).sum().backward()
        opt.step()
        step()
        opt.zero_grad()
        optt.zero_grad()
    return close(mt.weight, m.weight, 1e-3) or (mt.weight.cpu(), m.weight)


print("\nSUMMARY")
for n, r in results:
    print(f"{n:35s} {r[:200]}")
