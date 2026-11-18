import traceback
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


