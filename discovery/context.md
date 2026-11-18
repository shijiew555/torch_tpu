# Discovery context (plan phase, 2026-11-18 ~02:40Z)

## Environment facts

* TPU: one v5e chip (`tpu-info`). libtpu 0.0.50.
* Host: 24 CPUs, 47 GB RAM, 225 GB free disk.
* System Python 3.13: torch 2.14.1+cpu. torch_tpu was upgraded from 0.2.0 to
  0.3.0 (PyPI) in the plan phase. Smoke test on TPU passes.
* 0.3.0 is the newest release (closer to main). 0.2.0 had many bugs that are
  already fixed. `torch_tpu.__version__` does not exist; use
  `pip show torch_tpu`. Use system `python` for repros.
* upstream/main = afd6943a. 472 commits since 2026-11-01. The project moves
  fast. Before you report a bug, check `git log upstream/main` and the open PR
  list for a fix.
* Source build is a risk: issue #6508 says `--config=no_rbe` builds fail since
  the Bazel 9.3.1 upgrade. Workaround in the issue:
  `--spawn_strategy=sandboxed,local` after `--config=no_rbe`.
  A full wheel build compiles XLA locally (hours on 24 CPUs).
* Most Python code is in `src/torch_tpu/_internal/` (132 files: compiler
  backend, view decomposition, APIs). Python-only fixes are cheapest to test
  (overlay the changed file on the installed wheel). C++ op fixes need a build.

## User's prior work (GitHub `shijiew555`)

* No issues and no PRs in `google-pytorch/torch_tpu`. No duplicate risk.
* Other repos only (helion, vllm-omni, FastVideo, distserve, ...).

## What maintainers accept (evidence)

* External PRs land through Copybara: the PR is closed (not merged) and a
  commit appears on main. Example: #6312 (yashmehra-cmd) became 6d7003cb
  "Support cross-attention ... in splash_sdpa".
* Recent accepted fixes are small op-correctness and crash fixes, often with
  `[Ops][<op>]` title prefix: arange overflow, histc validation, remainder with
  inf divisor, complex->bool, pooling ceil_mode with dynamic shapes, max/min
  -inf start, leaky_relu backward at 0, bincount negative input.
* Issue reports are valued by themselves (CONTRIBUTING.md). Issue first for
  non-trivial changes.

## Open issues - all are assigned (treat as claimed, do not pick)

Every open issue has an assignee. Do not work on these topics:

* 6757 bind rank/chip after import. 6745 AI code review (infra).
* 6729 FSDP2 `_chunk_cat` compile time (open PR 6728 by HJSang).
* 6712 round(decimals) ties / fake_quantize (open PR 6758).
* 6677 nn.LSTM/GRU under torch.compile (open PR 6697 by apullin).
* 6675 strided view write/read aborts (open PR 6676 by apullin).
* 6663 huber_loss. 6630 view_as_complex offset (open PR 6631).
* 6593 F.linear keeps grad buffer. 6555 PJRT init IsInitialized (PR 6558).
* 6554 async compile drops options (PR 6557). 6508 no_rbe build.
* 6434 hardshrink/softshrink (PR 6674). 6397 RecordFunction slow path.
* 6393 compile cache ignores matmul precision. 6368 multi_margin_loss.
* 6338 median. 6304 SDPA routing perf. 6229 var/std bounded dynamism.
* 6216 stream API for D2H/H2D. 6118 logaddexp. 6113 extension loader msg.
* 6112 bounded dynamism deny list, clamp.Tensor. 6103 max_unpool2d.
* 6015 group_norm bf16/f32 dtype. 5829 cross_entropy compile time.
* 5725 gcd. 5679 mish. 5678 layer_norm under autocast bf16.
* 5674 view_decomposition storage-offset symbol. 5667 BatchNorm running stats.
* 5086 x[i] compiles per index. 5085 nonzero compile time.
* 5000 compile recompiles on storage offset. 4784 index_add_ wrong rows.
* 4413 log noise, torch.tpu memory APIs. 4412 distributed.environment missing.
* 4411 DDP + compile graph breaks. 4410 torch.tpu.precision graph break.
* 3169 MoE dispatch/combine lowering. CI/infra: 3909 3908 3502 2886 2384
  2366 851 850.

## Open PRs - topics to avoid (non-bot authors in bold)

* **6728** _chunk_cat kernel. **6697** LSTM/GRU compile. **6676** strided view
  aborts. **6674** hardshrink/softshrink. **6673** compound SymInt
  placeholders. **6631** view_as_complex. **6594** linear_backward split.
  **6576** P2pSendQueue test. **6558** PJRT client publish. **6557** async
  compile options.
* Bot (internal, in flight): 6759 SDPA StableHLO routing, 6758 round half even,
  6751 compile cache device assignment, 6748/6747 histc, 6746 XLA flash attn
  rewrite off, 6744 barrier unfused, 6741 is_current_stream_capturing, 6740
  symbolic repeat, 6738 memory stats before init, 6737 device-guard typed
  exceptions, 6724 optimization_barrier differentiable, 6701 fmax/fmin dyn,
  6698 _segment_reduce dyn, 6695 native_norm empty, 6665 _fused_dropout,
  6661 lu_solve dyn, 6603 _masked_scale, 6601 SDPA VMEM routing, 6592/6591/
  6588/6582/6579/6513/6512 TPU graph capture, 6584 N-D linear lowering, 6549
  split_with_sizes_copy, 6517 softmax dim validation, 6511/6509 flex_attention.
* Recently closed unmerged external: 6495 (apullin) max/min -inf - fixed
  internally (issue 6384 closed).

## Recently closed issues (fixed; do not re-report)

linspace int >2**24, fake_quantize ties, max/min -inf, SDPA dropout mask,
HF masking_utils host syncs, eight missing ops (smooth_l1_loss, huber_loss,
logaddexp, logaddexp2, nansum, std_mean, trace, addbmm - "fix on a branch"),
worker-thread tensor destroy abort, bincount, linalg_cross, scalar buffer
eviction, addmv validation, leaky_relu backward, avg_pool2d dyn backward,
splash_sdpa issues, _int_mm, compile forward copies weights, exponential_ 0,
dot validation, add/_foreach_add bf16 alpha, gelu fp32, SDPA broadcast mask,
view_decomposition cases (5672, 5671), BatchNorm momentum=None recompiles,
tied params untied by .to(), AOTAutogradCache keys, SDPA float32 precision,
OOM message, grouped_mm empty groups, compile(mode=...) fails, inductor backend
error, antialias interpolate, device_count before/after init, explicit device
index ignored, storage-offset specialization, boolean mask alignment,
unique(dim), cholesky_ex, SDPA attn_mask grad, fully masked row leak,
interpolate linear abort, ctc_loss abort, multinomial seed, avg_pool1d autograd,
all_reduce non-sum, bf16 cast >2^30, DataLoader fork hang, pip torch mismatch.
Use `gh issue list -R google-pytorch/torch_tpu --state all --search "<op>"`
for any candidate.
