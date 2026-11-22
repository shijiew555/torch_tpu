"""Common user mistakes on CPU vs TPU. Prints one line per case."""
import faulthandler, sys
import torch
import torch.nn.functional as F
faulthandler.enable()
faulthandler.dump_traceback_later(500, exit=True)
ONLY = sys.argv[1:] 

def mat(x):
  if isinstance(x, torch.Tensor): return x.cpu()
  if isinstance(x, (list, tuple)): return [mat(y) for y in x]
  return x

def run(fn, d):
  try:
    r = mat(fn(d))
    torch.accelerator.synchronize() if d != "cpu" else None
    s = repr(r)
    return "ok " + s.replace("\n", " ")[:150]
  except Exception as e:
    return f"{type(e).__name__}: " + str(e).replace("\n", " | ")[:300]

C = {}
def case(f):
  C[f.__name__] = f; return f

@case
def mixed_add(d): return torch.ones(3, device=d) + torch.ones(3)
@case
def mixed_add_scalar_cpu(d): return torch.ones(3, device=d) + torch.tensor(2.0)
@case
def mixed_matmul(d): return torch.ones(3, 3, device=d) @ torch.ones(3, 3)
@case
def mixed_index(d): return torch.arange(5., device=d)[torch.tensor([1, 2])]
@case
def mixed_where(d): return torch.where(torch.ones(3, device=d) > 0, torch.ones(3), torch.zeros(3, device=d))
@case
def index_select_oob(d): return torch.arange(5., device=d).index_select(0, torch.tensor([1, 7], device=d))
@case
def index_select_neg(d): return torch.arange(5., device=d).index_select(0, torch.tensor([-1], device=d))
@case
def gather_oob(d): return torch.arange(5., device=d).gather(0, torch.tensor([9], device=d))
@case
def embedding_oob(d): return F.embedding(torch.tensor([0, 10], device=d), torch.ones(4, 2, device=d))
@case
def embedding_neg(d): return F.embedding(torch.tensor([-1], device=d), torch.ones(4, 2, device=d))
@case
def adv_index_oob(d): return torch.arange(5., device=d)[torch.tensor([7], device=d)]
@case
def getitem_int_oob(d): return torch.arange(5., device=d)[7]
@case
def setitem_oob(d):
  x = torch.zeros(5, device=d); x[torch.tensor([9], device=d)] = 1.; return x
@case
def scatter_oob(d): return torch.zeros(5, device=d).scatter(0, torch.tensor([9], device=d), 1.0)
@case
def scatter_add_oob(d): return torch.zeros(5, device=d).scatter_add(0, torch.tensor([9], device=d), torch.ones(1, device=d))
@case
def index_add_oob(d): return torch.zeros(5, device=d).index_add(0, torch.tensor([9], device=d), torch.ones(1, device=d))
@case
def index_put_oob(d): return torch.zeros(5, device=d).index_put((torch.tensor([9], device=d),), torch.ones(1, device=d))
@case
def take_oob(d): return torch.arange(5., device=d).take(torch.tensor([9], device=d))
@case
def one_hot_oob(d): return F.one_hot(torch.tensor([0, 5], device=d), 3)
@case
def one_hot_neg(d): return F.one_hot(torch.tensor([-1], device=d), 3)
@case
def bincount_neg(d): return torch.bincount(torch.tensor([-1, 2], device=d))
@case
def cat_dtype(d): return torch.cat([torch.ones(2, device=d), torch.ones(2, device=d, dtype=torch.int64)])
@case
def cat_shape(d): return torch.cat([torch.ones(2, 3, device=d), torch.ones(2, 4, device=d)])
@case
def cat_ndim(d): return torch.cat([torch.ones(2, 3, device=d), torch.ones(3, device=d)])
@case
def cat_empty_list(d): return torch.cat([])
@case
def stack_shape(d): return torch.stack([torch.ones(2, device=d), torch.ones(3, device=d)])
@case
def view_noncontig(d): return torch.ones(3, 4, device=d).t().view(12)
@case
def view_bad_numel(d): return torch.ones(3, 4, device=d).view(5, 3)
@case
def reshape_bad(d): return torch.ones(3, 4, device=d).reshape(5, -1)
@case
def out_dtype_wrong(d): 
  o = torch.empty(3, device=d, dtype=torch.int64); return torch.add(torch.ones(3, device=d), 0.5, out=o)
@case
def out_device_wrong(d):
  o = torch.empty(3); return torch.add(torch.ones(3, device=d), 1, out=o)
@case
def out_device_wrong_mm(d):
  o = torch.empty(2, 2); return torch.mm(torch.ones(2, 2, device=d), torch.ones(2, 2, device=d), out=o)
@case
def out_dtype_wrong_mm(d):
  o = torch.empty(2, 2, device=d, dtype=torch.float64); return torch.mm(torch.ones(2, 2, device=d), torch.ones(2, 2, device=d), out=o)
@case
def matmul_shape(d): return torch.ones(2, 3, device=d) @ torch.ones(4, 2, device=d)
@case
def mm_shape(d): return torch.mm(torch.ones(2, 3, device=d), torch.ones(4, 2, device=d))
@case
def bmm_batch(d): return torch.bmm(torch.ones(2, 2, 3, device=d), torch.ones(3, 3, 2, device=d))
@case
def mm_dtype(d): return torch.mm(torch.ones(2, 2, device=d), torch.ones(2, 2, device=d, dtype=torch.float64))
@case
def mm_dtype_bf16(d): return torch.mm(torch.ones(2, 2, device=d), torch.ones(2, 2, device=d, dtype=torch.bfloat16))
@case
def linear_shape(d): return F.linear(torch.ones(2, 3, device=d), torch.ones(4, 5, device=d))
@case
def linear_dtype(d): return F.linear(torch.ones(2, 3, device=d, dtype=torch.bfloat16), torch.ones(4, 3, device=d))
@case
def ce_target_oob(d): return F.cross_entropy(torch.randn(2, 3, device=d), torch.tensor([0, 5], device=d))
@case
def ce_target_neg(d): return F.cross_entropy(torch.randn(2, 3, device=d), torch.tensor([0, -5], device=d))
@case
def ce_target_float_wrongshape(d): return F.cross_entropy(torch.randn(2, 3, device=d), torch.tensor([0., 1.], device=d))
@case
def ce_target_int32(d): return F.cross_entropy(torch.randn(2, 3, device=d), torch.tensor([0, 1], device=d, dtype=torch.int32))
@case
def ce_target_shape(d): return F.cross_entropy(torch.randn(2, 3, device=d), torch.tensor([0, 1, 2], device=d))
@case
def nll_target_oob(d): return F.nll_loss(torch.randn(2, 3, device=d), torch.tensor([0, 3], device=d))
@case
def ce_ignore_index_ok(d): return F.cross_entropy(torch.randn(2, 3, device=d), torch.tensor([0, -100], device=d))
@case
def bce_target_range(d): return F.binary_cross_entropy(torch.tensor([0.5], device=d), torch.tensor([2.0], device=d))
@case
def bce_input_range(d): return F.binary_cross_entropy(torch.tensor([1.5], device=d), torch.tensor([1.0], device=d))
@case
def mse_shape(d): return F.mse_loss(torch.ones(3, device=d), torch.ones(4, device=d))
@case
def conv_channels(d): return F.conv2d(torch.ones(1, 3, 8, 8, device=d), torch.ones(4, 2, 3, 3, device=d))
@case
def conv_ndim(d): return F.conv2d(torch.ones(3, 8, device=d), torch.ones(4, 3, 3, 3, device=d))
@case
def conv_kernel_too_big(d): return F.conv2d(torch.ones(1, 3, 2, 2, device=d), torch.ones(4, 3, 3, 3, device=d))
@case
def layernorm_shape(d): return F.layer_norm(torch.ones(2, 3, device=d), (4,))
@case
def bn_channels(d): return F.batch_norm(torch.ones(2, 3, device=d), torch.zeros(4, device=d), torch.ones(4, device=d))
@case
def softmax_dim(d): return torch.softmax(torch.ones(2, 3, device=d), dim=2)
@case
def sum_dim_oob(d): return torch.ones(2, 3, device=d).sum(dim=3)
@case
def transpose_oob(d): return torch.ones(2, 3, device=d).transpose(0, 2)
@case
def permute_bad(d): return torch.ones(2, 3, device=d).permute(0, 0)
@case
def expand_bad(d): return torch.ones(2, 3, device=d).expand(4, 3)
@case
def broadcast_bad(d): return torch.ones(2, 3, device=d) + torch.ones(4, device=d)
@case
def inplace_broadcast_bad(d):
  x = torch.ones(3, device=d); x.add_(torch.ones(2, 3, device=d)); return x
@case
def inplace_dtype_bad(d):
  x = torch.ones(3, device=d, dtype=torch.int64); x.add_(torch.ones(3, device=d) * 0.5); return x
@case
def leaf_inplace(d):
  x = torch.ones(3, device=d, requires_grad=True); x.add_(1); return x
@case
def backward_nonscalar(d):
  x = torch.ones(3, device=d, requires_grad=True); (x * 2).backward(); return x.grad
@case
def item_multi(d): return torch.ones(3, device=d).item()
@case
def numpy_call(d): return torch.ones(3, device=d).numpy()
@case
def int_tensor_requires_grad(d): return torch.ones(3, device=d, dtype=torch.int64).requires_grad_()
@case
def split_bad(d): return torch.ones(5, device=d).split([2, 2])
@case
def chunk_zero(d): return torch.ones(5, device=d).chunk(0)
@case
def narrow_oob(d): return torch.ones(5, device=d).narrow(0, 3, 4)
@case
def select_oob(d): return torch.ones(5, device=d).select(0, 9)
@case
def repeat_fewer(d): return torch.ones(2, 3, device=d).repeat(2)
@case
def randint_bad(d): return torch.randint(5, 2, (3,), device=d)
@case
def multinomial_neg(d): return torch.multinomial(torch.tensor([0.5, -0.1], device=d), 1)
@case
def multinomial_zero(d): return torch.multinomial(torch.tensor([0., 0.], device=d), 1)
@case
def multinomial_noreplace(d): return torch.multinomial(torch.tensor([0.5, 0.5], device=d), 3)
@case
def normal_neg_std(d): return torch.normal(torch.zeros(3, device=d), -torch.ones(3, device=d))
@case
def bernoulli_p_bad(d): return torch.bernoulli(torch.tensor([1.5], device=d))
@case
def empty_neg(d): return torch.empty(-1, device=d)
@case
def zeros_neg(d): return torch.zeros(2, -3, device=d)
@case
def full_neg(d): return torch.full((-2,), 1.0, device=d)
@case
def arange_bad_step(d): return torch.arange(0, 5, 0, device=d)
@case
def arange_wrong_sign(d): return torch.arange(0, 5, -1, device=d)
@case
def linspace_neg(d): return torch.linspace(0, 1, -1, device=d)
@case
def tensor_to_bad_dtype_cmp(d): return torch.ones(3, device=d) < torch.ones(3, device=d, dtype=torch.complex64)
@case
def max_empty(d): return torch.ones(0, device=d).max()
@case
def argmax_empty(d): return torch.ones(0, device=d).argmax()
@case
def min_empty_dim(d): return torch.ones(0, 3, device=d).min(dim=0)
@case
def mean_int(d): return torch.ones(3, device=d, dtype=torch.int64).mean()
@case
def std_int(d): return torch.ones(3, device=d, dtype=torch.int64).std()
@case
def topk_dim(d): return torch.ones(3, device=d).sort(dim=2)
@case
def repeat_interleave_neg(d): return torch.repeat_interleave(torch.ones(2, device=d), torch.tensor([1, -1], device=d))
@case
def masked_fill_mask_dtype(d): return torch.ones(3, device=d).masked_fill(torch.ones(3, device=d), 0.)
@case
def masked_fill_mask_shape(d): return torch.ones(3, device=d).masked_fill(torch.ones(4, device=d, dtype=torch.bool), 0.)
@case
def bool_mask_shape(d): return torch.ones(3, device=d)[torch.ones(4, device=d, dtype=torch.bool)]
@case
def index_float(d): return torch.ones(3, device=d)[torch.tensor([0.], device=d)]
@case
def index_select_float_idx(d): return torch.ones(3, device=d).index_select(0, torch.tensor([0.], device=d))
@case
def gather_int32_idx(d): return torch.ones(3, device=d).gather(0, torch.tensor([0], dtype=torch.int32, device=d))
@case
def embedding_float_idx(d): return F.embedding(torch.tensor([0.], device=d), torch.ones(4, 2, device=d))
@case
def sdpa_shape(d): return F.scaled_dot_product_attention(torch.ones(1, 2, 4, 8, device=d), torch.ones(1, 2, 4, 6, device=d), torch.ones(1, 2, 4, 8, device=d))
@case
def sdpa_dtype(d): return F.scaled_dot_product_attention(torch.ones(1, 2, 4, 8, device=d), torch.ones(1, 2, 4, 8, device=d, dtype=torch.bfloat16), torch.ones(1, 2, 4, 8, device=d))
@case
def interpolate_bad(d): return F.interpolate(torch.ones(1, 1, 4, device=d), size=(2, 2))
@case
def pad_bad(d): return F.pad(torch.ones(2, 3, device=d), (1, 2, 3))
@case
def pad_reflect_too_big(d): return F.pad(torch.ones(1, 2, 3, device=d), (4, 4), mode="reflect")
@case
def maxpool_too_big(d): return F.max_pool2d(torch.ones(1, 1, 2, 2, device=d), 3)
@case
def avgpool_pad_too_big(d): return F.avg_pool2d(torch.ones(1, 1, 4, 4, device=d), 2, padding=2)
@case
def to_int_dtype_nan(d): return torch.tensor([float('nan')], device=d).long()
@case
def clamp_min_gt_max(d): return torch.arange(5., device=d).clamp(3, 1)
@case
def div_int_zero(d): return torch.tensor([1], device=d) // torch.tensor([0], device=d)
@case
def remainder_int_zero(d): return torch.tensor([1], device=d) % torch.tensor([0], device=d)
@case
def cholesky_nonpd(d): return torch.linalg.cholesky(-torch.eye(2, device=d))
@case
def inv_singular(d): return torch.linalg.inv(torch.zeros(2, 2, device=d))
@case
def solve_singular(d): return torch.linalg.solve(torch.zeros(2, 2, device=d), torch.ones(2, device=d))
@case
def inv_nonsquare(d): return torch.linalg.inv(torch.ones(2, 3, device=d))
@case
def det_nonsquare(d): return torch.linalg.det(torch.ones(2, 3, device=d))

for n, f in C.items():
  if ONLY and n not in ONLY: continue
  print(f">>> {n}", flush=True)
  c = run(f, "cpu")
  t = run(f, "tpu")
  same = "SAME" if c.split(":")[0].split(" ")[0] == t.split(":")[0].split(" ")[0] else "DIFF"
  print(f"{same} {n}\n  cpu: {c}\n  tpu: {t}", flush=True)
