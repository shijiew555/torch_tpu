import faulthandler, warnings; faulthandler.dump_traceback_later(400, exit=True)
import torch, torch_tpu  # noqa
d="tpu"
def t(name, f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = f()
        print(f"[{name}] OK: {str(r)[:600]!s}  warns={[str(i.message)[:100] for i in w]}")
    except Exception as e:
        print(f"[{name}] EXC {type(e).__name__}: {str(e)[:400]}")
x = torch.randn(2000, 300)
t("repr_large_match", lambda: repr(x.to(d)).replace("device='tpu:0'","").replace(", ","") == repr(x).replace(", ",""))
t("repr_large", lambda: repr(x.to(d))[-200:])
t("repr_bf16", lambda: (repr(torch.randn(3).bfloat16().to(d)), repr(torch.randn(3).bfloat16())))
t("repr_complex", lambda: repr(torch.randn(3, dtype=torch.complex64).to(d)))
t("repr_empty", lambda: repr(torch.empty(0,3).to(d)))
t("repr_noncontig", lambda: repr(torch.arange(12.).reshape(3,4).to(d).t()[1:, ::2]))
t("repr_int_bool", lambda: (repr(torch.tensor([1,2]).to(d)), repr(torch.tensor([True]).to(d))))
t("repr_inf_nan", lambda: repr(torch.tensor([float('inf'), float('nan'), 1e-8, 1e10]).to(d)))
t("repr_scalar", lambda: repr(torch.tensor(3.5).to(d)))
t("repr_requires_grad", lambda: repr((torch.ones(2,device=d,requires_grad=True)*2)))
t("repr_fp8", lambda: repr(torch.randn(3).to(torch.float8_e4m3fn).to(d)))
t("repr_f64", lambda: repr(torch.tensor([1/3], dtype=torch.float64).to(d)))
def po():
    torch.set_printoptions(precision=2, sci_mode=False, threshold=5, edgeitems=1)
    r = repr(torch.arange(10.).to(d)/3); torch.set_printoptions(profile="default"); return r
t("set_printoptions", po)
t("assert_close_tt", lambda: torch.testing.assert_close(torch.ones(3,device=d), torch.ones(3,device=d)))
t("assert_close_fail_msg", lambda: torch.testing.assert_close(torch.ones(3,device=d), torch.tensor([1.,2,1],device=d)))
t("assert_close_mixed", lambda: torch.testing.assert_close(torch.ones(3,device=d), torch.ones(3)))
t("assert_close_mixed_nodev", lambda: torch.testing.assert_close(torch.ones(3,device=d), torch.ones(3), check_device=False))
t("numpy", lambda: torch.ones(2,device=d).numpy())
t("numpy_force", lambda: torch.ones(2,device=d).numpy(force=True))
t("np_array", lambda: __import__("numpy").asarray(torch.ones(2,device=d)))
t("sparse_coo_to", lambda: torch.eye(3).to_sparse().to(d))
t("sparse_csr_to", lambda: torch.eye(3).to_sparse_csr().to(d))
t("nested_jagged", lambda: torch.nested.nested_tensor([torch.ones(2,4), torch.ones(3,4)], layout=torch.jagged, device=d).sin().values().cpu().shape)
t("nested_strided", lambda: torch.nested.nested_tensor([torch.ones(2,4), torch.ones(3,4)], device=d).sin().to_padded_tensor(0).cpu().shape)
def det():
    torch.use_deterministic_algorithms(True)
    try:
        out = []
        a = torch.zeros(5, device=d); idx = torch.tensor([0,0,1]).to(d)
        out.append(a.index_put((idx,), torch.ones(3,device=d), accumulate=True).cpu().tolist())
        out.append(torch.zeros(5,device=d).scatter_add(0, idx, torch.ones(3,device=d)).cpu().tolist())
        out.append(torch.arange(4.,device=d).cumsum(0).cpu().tolist())
        out.append(torch.zeros(5,device=d).index_add(0, idx, torch.ones(3,device=d)).cpu().tolist())
        e = torch.nn.Embedding(5,2).to(d); e(idx).sum().backward(); out.append(e.weight.grad.cpu()[0].tolist())
        out.append(torch.bincount(idx).cpu().tolist())
        out.append(torch.nn.functional.interpolate(torch.ones(1,1,4,4,device=d,requires_grad=True), scale_factor=2, mode="bilinear").sum().item())
        out.append(torch.empty(3, device=d).isnan().any().item())
        return out
    finally:
        torch.use_deterministic_algorithms(False)
t("deterministic_ops", lambda: det())
t("backends", lambda: dict(cudnn=torch.backends.cudnn.is_available(), mps=torch.backends.mps.is_available(),
   tf32=torch.backends.cuda.matmul.allow_tf32, fp32prec=torch.backends.fp32_precision if hasattr(torch.backends,"fp32_precision") else None,
   sdp_flash=torch.backends.cuda.flash_sdp_enabled(), cur_acc=torch.accelerator.current_accelerator(),
   amp=torch.amp.is_autocast_available("tpu"), get_device_module=torch.get_device_module("tpu").__name__))
t("is_pinned_tpu_tensor", lambda: torch.ones(2,device=d).is_pinned())
t("pin_memory_cpu", lambda: (torch.ones(2).pin_memory().is_pinned()))
t("tensor_device_index", lambda: (torch.ones(1,device="tpu").device, torch.ones(1,device="tpu").get_device()))
t("float_on_tpu", lambda: float(torch.tensor(2.5,device=d)))
t("bool_on_tpu", lambda: bool(torch.tensor(True,device=d)))
t("dlpack", lambda: torch.from_dlpack(torch.ones(2,device=d)))
t("hash_set", lambda: len({torch.ones(1,device=d), torch.ones(1,device=d)}))
t("storage", lambda: torch.ones(3,device=d).untyped_storage().nbytes())
t("data_ptr", lambda: torch.ones(3,device=d).data_ptr() != 0)
t("share_memory_", lambda: torch.ones(3,device=d).share_memory_())
t("pickle", lambda: __import__("pickle").loads(__import__("pickle").dumps(torch.ones(2,device=d))).device)
