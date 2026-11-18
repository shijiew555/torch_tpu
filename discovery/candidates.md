# Discovery candidates - round 1 (merge phase)

Status: done, 2026-11-18T05:20Z.
Author: explorer-controller (merge phase).
Sources: E1, E2, E3, E4, E6 reports in this folder. E5 was skipped.
Verification logs and probes: `/root/torch_tpu/.claude/tmp/merge/`.
upstream/main is now e411f09d (1 commit after afd6943a; it does not touch any candidate).
The base build (`tmp/controller/base_build2.log`) builds afd6943a.

## Result

* 5 candidates. All 5 are reproduced again by me on TPU (0.3.0), today.
* All root-cause lines are present on upstream/main e411f09d.
* No overlap found for any candidate (issues and PRs, all states).
* 2 are Python-only (testable now). 3 are C++ (1-10 lines each, need the base build).
* C++ risk: the base build is at 18,523 / 20,617 at 05:13Z. ETA 07:30-08:30Z. An incremental rebuild is not proven yet. A C++ candidate can miss the 08:32Z wrap-up.

| Rank | Title | Kind | Validation risk | Issue first? |
| --- | --- | --- | --- | --- |
| 1 | Make `torch.autocast("tpu")` default to bfloat16 | Python | Low | Yes (short) |
| 2 | Use HIGHEST precision in unfold backward and col2im | C++ | Medium-high | Yes (short) |
| 3 | Keep output aliases in TPU-compiled inference graphs | Python | Medium | Yes |
| 4 | Accept `torch.unique(sorted=False)` on TPU | C++ | Medium-high | No (trivial) |
| 5 | Stop DeprecationWarnings on non-blocking H2D copies | C++ | Medium-high | No (trivial) |

Suggested order of work: 1 and 3 now (Python, overlay). 2, 4, 5 as one queue after the build finishes (all are tiny; build once, test all).
If time is short, do 2 before 4 and 5 (higher value).

## 1. Make `torch.autocast("tpu")` default to bfloat16

* Rank 1. Common CUDA-port pattern silently runs in float16 and overflows; docs say bfloat16; Python-only.
* Evidence (my re-run, `tmp/merge/autocast.log`):
  * Command: `cd .claude/tmp/E1 && flock /root/tpu.lock timeout 300 python autocast_default.py`
  * Output:

    ```text
    torch.tpu._get_autocast_dtype(): torch.bfloat16
    torch.get_autocast_dtype('tpu'): torch.float16
    mm dtype under torch.autocast('tpu'): torch.float16
    mm dtype under torch.autocast('cpu'): torch.bfloat16
    300x300 sum of 4 under autocast('tpu'): inf
    ```

  * Expected: bfloat16 and 360000.0.
    `docs/api_reference.md` says "`bfloat16` by default on TPU".
* Confirmed facts:
  * `torch.autocast.__init__` reads `torch.get_autocast_dtype("tpu")` (torch 2.14 `amp/autocast_mode.py:233`). This is the C++ per-thread value. Core sets float16 for PrivateUse1.
  * torch_tpu sets only a Python class attribute (`src/torch_tpu/_internal/device/_device_module.py:414`, `_autocast_dtype = torch.bfloat16`). No code in `csrc` or `src` calls `set_autocast_dtype`.
  * Probe `tmp/merge/autocast_fix_probe.py`: after `torch.set_autocast_dtype("tpu", torch.bfloat16)`, the main-thread matmul under `torch.autocast("tpu")` is bfloat16.
  * Same probe: a new Python thread gets float16 again (value is thread-local).
  * `tests/amp_test.py:34` checks only `torch.tpu._get_autocast_dtype()`, so CI does not see the bug.
* Hypotheses:
  * Autograd engine threads copy the thread-local state from the caller, so backward is covered. Not checked.
  * A full per-thread fix needs a PyTorch core hook for a PrivateUse1 default dtype. Out of scope; mention in the issue.
* Proposed scope: set the PrivateUse1 autocast dtype to bfloat16 when torch_tpu initializes (one call in the import or device-module init path). Optionally make `_set_autocast_dtype` also call `torch.set_autocast_dtype`. Add tests. No C++ change. No change to explicit `dtype=` behavior.
* Validation plan:
  * Test in `tests/amp_test.py`: `torch.get_autocast_dtype("tpu") == torch.bfloat16`; matmul under `torch.autocast("tpu")` returns bfloat16.
  * Run on the E6 overlay: `PYTHONPATH=$E/ov flock /root/tpu.lock timeout 900 python3 -m tests.amp_test`.
  * `BASE_SHA=upstream/main nox -s lint`.
* Project benefit: matches the docs; avoids fp16 overflow (inf/NaN) for users who copy CUDA code; bf16 is the TPU-native type.
* Issue first: yes. It is a default-behavior change, so a short issue lets maintainers confirm the intent (docs and class attribute already say bf16).
* Overlap: none. Searched "autocast dtype", "autocast float16", "autocast default dtype". Related only: #5678 and PR 5691 (layer_norm under autocast).

## 2. Use HIGHEST precision in unfold backward and col2im (`F.fold`)

* Rank 2. Silent wrong float32 gradients (bf16-level, 1e-2 abs error) for `Tensor.unfold`, `F.unfold`/`nn.Unfold` backward and `F.fold`; C++ ~10 lines per file.
* Evidence (my re-run, `tmp/merge/repro_unfold_grad.py.log`, `repro_fold.py.log`):
  * Commands: `cd .claude/tmp/E3 && flock /root/tpu.lock timeout 150 python repro_unfold_grad.py` (and `repro_fold.py`).
  * Output:

    ```text
    tpu f32 max err: 0.010264694690704346
    tpu f32 bf16-rounded-input err: 0.010264694690704346
    F.fold (col2im) tpu f32 max err: 0.018382906913757324
    F.fold cpu f32 max err: 9.969808161258698e-07
    ```

  * `torch.set_float32_matmul_precision("highest")` is set in both repros and does not help.
* Confirmed facts:
  * `csrc/ops/unfold/unfold_aten_kernels.cc:129` and `csrc/ops/col2im/col2im_aten_kernels.cc:198` pass `/*precision_config=*/{}` to a `stablehlo::Convolution` with a 0/1 weight. These are the only two empty precision configs in `csrc` (`git grep` on upstream/main).
  * The TPU error equals the CPU error with the cotangent rounded to bf16, so the input is rounded to bf16.
  * Precedent on main: `csrc/ops/gather/gather.cc:209-218` sets `HIGHEST` for f32 with `// EXPLICIT_PRECISION_OK=f32_gather` on a one-hot dot. The lint rule `CheckExplicitPrecisionUsage` allows this marker.
* Hypotheses: `HIGHEST` gives exact results (0/1 weights, sums only). The compile-time and runtime cost is small. Measure both.
* Proposed scope: in both kernels, set the precision config to `{HIGHEST, DEFAULT}` (or HIGHEST for both) when the element type is f32, following the gather.cc pattern. Add tests. Do not touch convolution or matmul lowerings.
* Validation plan:
  * Tests in `tests/ops_unit_test.py`: f32 `Tensor.unfold` backward, `F.unfold` backward and `F.fold` with `randn` inputs, compare with CPU float64, atol about 1e-5 (now 1e-2).
  * Build after the base build, install the wheel in a venv, run the test under the lock (E6 recipe). Quick timing of `F.fold` before/after.
  * `BASE_SHA=upstream/main nox -s lint`.
* Project benefit: correct gradients for patch-extraction and sliding-window layers in float32 training.
* Validation risk: medium-high. Needs the C++ build (ETA 07:30-08:30Z, incremental rebuild unproven).
* Issue first: yes, short. It changes numerics, and the lint rule asks for a reason for explicit precision.
* Overlap: none. Searched "unfold", "col2im", "fold precision", "precision_config". Old closed PR 3809 (conv precision) did not touch these files.

## 3. Keep output aliases in TPU-compiled inference graphs

* Rank 3. Silent wrong results: a compiled function that returns a tensor and its views loses the alias when no output needs grad; Python-only, but a medium-size change.
* Evidence (my re-run, `tmp/merge/alias.log`):
  * Command: `cd .claude/tmp/E2 && flock /root/tpu.lock timeout 300 python alias_repro.py`
  * Output (TPU rows):

    ```text
    tpu backend=None      requires_grad=False shared=True  b[0,:3]=[100.0, 102.0, 104.0] ...
    tpu backend=tpu       requires_grad=False shared=False b[0,:3]=[0.0, 2.0, 4.0] ...
    tpu backend=tpu       requires_grad=True  shared=True  b[0,:3]=[100.0, 102.0, 104.0] ...
    ```

  * CPU eager, `aot_eager` and inductor keep the alias (E2).
* Confirmed facts:
  * `src/torch_tpu/_internal/export/export.py` `_process_fx_outputs` dedups outputs only on equal data_ptr and full layout (`_TensorDedupeKey`). A view with another layout becomes a separate output buffer. Same code on upstream/main e411f09d.
  * AOTAutograd regenerates aliases only for outputs that require grad (E2 code reading; the `requires_grad=True` row agrees).
* Hypotheses: a Python fix can emit only the base and rebuild views with `as_strided` after reconstruction. The views-only case (base not returned) needs the base as a hidden extra output. Dynamic shapes need care.
* Proposed scope: in `_process_fx_outputs` and the output reconstruction path, detect outputs that share storage with another output and rebuild them as views. Add tests. No C++ change.
* Validation plan:
  * Compile test: return `(y, y.view(...), y[1:])` with no grad, mutate `y`, check views see it and share `untyped_storage()`. Also the views-only case. Compare with `aot_eager`.
  * Run on the E6 overlay under the lock. Run existing compile tests that touch output dedup (for example `tests/compile/*` that use `_process_fx_outputs`).
* Project benefit: matches eager and inductor semantics; removes a silent correctness bug; saves one device buffer per aliased output.
* Validation risk: medium. Python-only, but the change is 50-100 lines in a central path and may conflict with in-flight compile work.
* Issue first: yes. Non-trivial design (where to rebuild views); maintainers may prefer another shape.
* Overlap: none. Searched "alias outputs", "aliasing compile", "output aliasing". PR 6203 (open, write-back of in-place writes to inputs) and PR 6250 (closed, passthrough elision) are different topics, but touch the same area; mention them in the issue.

## 4. Accept `torch.unique(sorted=False)` on TPU

* Rank 4. A common argument raises on TPU; CPU and CUDA accept it; C++ 2-line fix with an in-file precedent.
* Evidence (my re-run, `tmp/merge/repro_unique.py.log`):
  * Command: `cd .claude/tmp/E3 && flock /root/tpu.lock timeout 150 python repro_unique.py`
  * Output: `cpu: tensor([1, 2, 3])`, then `RuntimeError: unique2(): sorted=False is not yet supported`.
* Confirmed facts:
  * `csrc/ops/unique/unique_aten_kernels.cc` `AtenUnique2`: `TT_CHECK_THROW(sorted, ...) << "sorted=False is not yet supported"`.
  * `AtenUniqueDim` in the same file ignores `sorted` ("does not affect SHLO") and always sorts.
* Hypotheses: CUDA `_unique2` also always sorts, so sorted output is valid for `sorted=False`. Check the CUDA source in the PR description.
* Proposed scope: remove the check; pass `true` to `GetUniqueCount` and `Unique2`. Add tests. Nothing else.
* Validation plan: test `torch.unique(x, sorted=False)` with `return_inverse` and `return_counts` for int64, f32, bf16 against CPU sorted output, in `tests/ops_unit_test.py`. Needs the C++ build.
* Project benefit: removes an error for library and user code that passes `sorted=False` for speed.
* Validation risk: medium-high (C++ build).
* Issue first: no. Trivial fix.
* Overlap: none. Searched "unique sorted", "unique". Closed issue 4159 was `unique(dim=...)`.

## 5. Stop DeprecationWarnings on non-blocking H2D copies of unpinned tensors

* Rank 5. Every `x.to("tpu", non_blocking=True)` of an unpinned tensor emits two torch 2.14 DeprecationWarnings (an error under `-W error`); C++ 1-line fix. Impact is low, but the fix is safe and the change is very likely to be merged.
* Evidence (my re-run, `tmp/merge/pinwarn.log`):
  * Command: `cd .claude/tmp/E4 && flock /root/tpu.lock timeout 300 python repro_pin_warn.py`
  * Output:

    ```text
    DeprecationWarning: The argument 'device' of Tensor.pin_memory() is deprecated. ...
    DeprecationWarning: The argument 'device' of Tensor.is_pinned() is deprecated. ...
    tensor([1., 1., 1., 1.])
    ```

* Confirmed facts:
  * `csrc/ops/copy_from/cpu_to_tpu.cc:125`: `effective_src = effective_src.pin_memory(dest.device());`.
  * Probe `tmp/merge/pin_probe.py`: `torch.ones(4).pin_memory()` (no device) gives a tensor that the TPU copy path treats as pinned. A non-blocking copy of it under `warnings.simplefilter("error")` passes. So `pin_memory()` with no argument gives TPU-pinned memory.
* Hypotheses: Python only shows DeprecationWarning for code in `__main__` by default, so library calls (HF, accelerate) are mostly silent. Pytest shows it.
* Proposed scope: call `pin_memory()` with no argument. Add a test. Nothing else.
* Validation plan: test a non-blocking H2D copy of an unpinned tensor under `warnings.catch_warnings()` + `simplefilter("error")`, check the value. Needs the C++ build.
* Project benefit: clean logs and `-W error` test suites for normal training loops.
* Validation risk: medium-high (C++ build).
* Issue first: no. Trivial fix.
* Overlap: none. Searched "pin_memory", "pin_memory deprecated", "non_blocking". PR 6449 (merged, D2H pageable) is a different path.

## Considered but not chosen

* `torch.Stream("tpu")` with no index raises (E4 F1, reproduced): real, C++ ~5 lines, but low use and open bot PR 6737 rewrites the same functions in `tpu_hooks.cc`. Revisit after 6737 lands.
* `adaptive_max_pool2d` backward not implemented (E3 F3, reproduced): blocks training, but needs a new C++ kernel; too large to validate before 08:32Z.
* `upsample_trilinear3d`, `kthvalue` missing (E3 F3): new C++ kernels; same reason.
* `torch.compile` default is `dynamic=False`, but an error message and `docs/dynamic_shapes.md` say `torch.compile(model)` is bounded dynamism (E1 F4): real wording bug, low impact; could be a small docs PR later, or a note in an issue.
* `torch.cond` / `while_loop` fail under `backend="tpu"` (E2 F2): feature gap; full fix needs C++ HOP lowering; a better error alone is low value.
* `.item()` slice length with `capture_scalar_outputs` gives an internal error (E2 F3): non-default flag, known-hard area (#5085, #6112).
* HF `Trainer(torch_compile=True)` cannot use the TPU backend (E1 F2): fix belongs in transformers and accelerate, not torch_tpu.
* accelerate 1.15.0 does not detect TPU (E1 F3): fixed in accelerate 1.16.0.
* Error of a failed lazy op shows at an unrelated `model.to` (E1 F5): cause is a hypothesis; better as a comment on #5678.
* f32 `log1p` less accurate on TPU (E3 hypothesis): likely in XLA/libtpu, not torch_tpu.
