import sys
import faulthandler
faulthandler.dump_traceback_later(600, exit=True)
import torch
import torch_tpu  # noqa: F401
exec(open("/root/torch_tpu/.claude/tmp/E2/sweep_common.py").read())


@case("alias_outputs_inference")
def _():
    def f(x):
        y = x * 2
        return y, y.view(2, 8)
    a, b = C(f)(torch.arange(16.0).view(4, 4).to(D))
    a.add_(100)
    return close(b, (torch.arange(16.0) * 2 + 100).view(2, 8)) or b[0, :3].cpu()


@case("alias_outputs_training")
def _():
    def f(x):
        y = x * 2
        return y, y.view(2, 8)
    xt = torch.arange(16.0).view(4, 4).to(D).requires_grad_()
    a, b = C(f)(xt)
    sh = a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    return sh or "no shared storage in training mode"


@case("alias_two_views_of_intermediate")
def _():
    def f(x):
        y = x * 2
        return y[0], y[1:]
    a, b = C(f)(torch.arange(16.0).view(4, 4).to(D))
    sh = a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    return sh or "no shared storage"


@case("cond_static_pred")
def _():
    def f(x, p: bool):
        return torch.cond(p, lambda t: t + 1, lambda t: t - 1, (x,))
    x = torch.randn(4)
    return close(C(f)(x.to(D), True), x + 1)


@case("cond_tensor_pred")
def _():
    def f(x):
        return torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))
    x = torch.randn(4).abs()
    return close(C(f)(x.to(D)), x.sin())


@case("cond_eager_tpu")
def _():
    def f(x):
        return torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))
    x = torch.randn(4).abs()
    return close(f(x.to(D)), x.sin())


@case("while_loop_compiled")
def _():
    from torch._higher_order_ops.while_loop import while_loop
    def f(i, x):
        return while_loop(lambda i, x: i < 3, lambda i, x: (i + 1, x * 2), (i, x))
    i = torch.tensor(0)
    x = torch.ones(3)
    out = C(f)(i.to(D), x.to(D))
    return close(out[1], x * 8)


@case("compiler_disable_inside")
def _():
    @torch.compiler.disable
    def g(x):
        return x.cpu().numpy().sum()
    def f(x):
        return x * float(g(x))
    x = torch.randn(4)
    return close(C(f)(x.to(D)), x * float(x.sum()))


@case("item_capture_scalar")
def _():
    torch._dynamo.config.capture_scalar_outputs = True
    try:
        def f(x):
            s = x.sum().item()
            return x * s + 1
        x = torch.randn(4)
        return close(C(f)(x.to(D)), x * x.sum() + 1)
    finally:
        torch._dynamo.config.capture_scalar_outputs = False


@case("int_item_shape")
def _():
    torch._dynamo.config.capture_scalar_outputs = True
    try:
        def f(x, n):
            k = n.item()
            return x[:k].sum()
        x = torch.randn(8)
        return close(C(f)(x.to(D), torch.tensor(5).to(D)), x[:5].sum())
    finally:
        torch._dynamo.config.capture_scalar_outputs = False


print("\nSUMMARY")
for n, r in results:
    print(f"{n:35s} {r[:200]}")
