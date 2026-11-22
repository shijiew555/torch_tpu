# Discovery candidates - session 3 (merge phase)

Status: done, 2026-11-22T23:35Z.
Author: explorer controller (merge phase).
Sources: `S3-E1.md` to `S3-E4.md` in this folder.
My verification logs: `$COORD/scratch/merge-s3/` (`ckpt.log`, `norm.log`,
`empty.log`).
Wheel: `afd6943a` (discovery venv). Root causes read on upstream/main
`82d0a494`.

## Result

* The user decided that the four session 1-2 branches are PRs 1-4.
  So we need one task for PR 5 and two backups.
* Top pick: fix eager `torch.utils.checkpoint` so that it restores the TPU
  RNG state. Python-only, about 10 lines. Silent wrong gradients.
* Backup 1: validate `normalized_shape` and the weight/bias shapes in
  `layer_norm`. C++, about 20 lines. Silent zeros today.
* Backup 2: indexing an empty tensor with an empty index aborts the
  process. C++, about 10-20 lines.
* I re-ran all three on TPU today. All three reproduce.
* The root-cause code of all three is unchanged on upstream/main.
* `gh` searches (issues and PRs, all states) found no overlap for any of
  the three.

| Rank | Title | Kind | Finish by 04:40Z from a 00:30Z start? |
| --- | --- | --- | --- |
| 1 | Restore the TPU RNG state in eager activation checkpointing | Python | Yes, high confidence (about 1.5 h) |
| 2 | Validate input, weight and bias shapes in `layer_norm` | C++ | Yes, likely (about 2.5 h with a 20-40 min build) |
| 3 | Return an empty result when indexing an empty dimension | C++ | Yes, likely (about 2.5 h with a 20-40 min build) |

## 1. Restore the TPU RNG state in eager activation checkpointing

* Rank 1. Silent wrong gradients for every eager model that uses
  activation checkpointing with dropout (HF
  `gradient_checkpointing_enable()`, torchtitan AC). Python-only fix.
* Evidence (my re-run, `scratch/merge-s3/ckpt.log`):
  * Command: `flock /root/tpu.lock timeout 300 <venv python>
    $COORD/scratch/S3-E1/ckpt_dropout.py`
  * Output:

    ```text
    hasattr(torch.tpu, '_initialized'): False
    cpu use_reentrant=True: grad diff vs no checkpoint = 0.0
    cpu use_reentrant=False: grad diff vs no checkpoint = 0.0
    tpu use_reentrant=True: grad diff vs no checkpoint = 1.5137939453125
    tpu use_reentrant=False: grad diff vs no checkpoint = 1.5137939453125
    ```

  * Expected: `0.0` on TPU, as on CPU.
  * S3-E1: with `torch.tpu._initialized = True` set by hand, all four
    lines give `0.0`. A two-layer `nn.TransformerEncoderLayer(dropout=0.1)`
    gives a relative grad diff of 0.705 on TPU and 0 on CPU.
* Confirmed facts:
  * `torch/utils/checkpoint.py` (torch 2.16 nightly, lines 272, 1983,
    2077) saves and restores device RNG state only if
    `getattr(device_module, "_initialized", False)` is true.
  * `torch.tpu` has no `_initialized` (re-run output above).
    `git grep "_initialized\b" upstream/main -- src` finds only
    `is_initialized`. No commit in `afd6943a..upstream/main` adds it.
  * `torch.compile(backend="tpu")` with checkpoint is correct (S3-E1).
    The compiled path uses graphsafe RNG (PR 5067). Only eager is wrong.
* Hypotheses:
  * A metaclass property on `_DeviceModuleMeta`
    (`src/torch_tpu/_internal/device/_device_module.py`) is reached by
    `getattr(torch.tpu, "_initialized")` through `private_fallback` in
    `public_api.py`. The worker must confirm this.
  * The property must call the backend `_is_initialized()`, not the
    `@experimental` `is_initialized()`, to avoid a warning on each
    checkpoint call.
  * The API scan tests (`tt_apis.yaml`, `scan_tt_apis`) may list the new
    private name. Check them.
* Proposed scope: add `torch.tpu._initialized` (true after the TPU
  runtime initializes) and a test. No change to RNG code, to the compiled
  path, or to PyTorch.
* Validation plan:
  * New test: eager `checkpoint` with dropout, `use_reentrant` True and
    False, grads equal to a run without checkpoint, same seed. Put it
    next to the activation checkpointing tests (for example a new eager
    case beside `tests/compile/upstream_activation_checkpointing_test.py`,
    or the device module tests). Follow the `tt-write-tests` skill.
  * Run it on the E6 overlay (Python-only) under the TPU lock: fail
    before, pass after.
  * `BASE_SHA=upstream/main nox -s lint`.
* Project benefit: correct gradients for a very common memory-saving
  training setup. Today the bug gives no error and no warning.
* Issue first: yes, a short one. It is a silent correctness bug, and an
  issue makes it easy to find. The fix is small, so the PR can follow at
  once.
* Overlap: none. Searched "checkpoint dropout", "checkpoint rng",
  "gradient checkpointing", "_initialized". Related only: PR 5067
  (compiled AC RNG, merged), PR 5666 (ported compiled AC tests), PR 6550.
* Finish estimate: Python-only. No build. About 1-1.5 h including review.

## 2. Validate input, weight and bias shapes in `layer_norm`

* Rank 2. A wrong `normalized_shape` gives all zeros with no error.
  Other wrong shapes give an MLIR error text. This matches the
  "[Ops][x] Validate ..." pattern that maintainers merge often.
* Evidence (my re-run, `scratch/merge-s3/norm.log`):
  * Command: `flock /root/tpu.lock timeout 300 <venv python> -c "import
    torch_tpu; exec(open('$COORD/scratch/S3-E3/repro_norm_shape.py').read())"`
  * Output:

    ```text
    layer_norm_shape
      cpu: RuntimeError: Given normalized_shape=[4], expected input with shape [*, 4], but got input of size[2, 3]
      tpu: ok shape=(2, 3) sum=0.0000
    layer_norm_shape_same_numel
      cpu: RuntimeError: Given normalized_shape=[3, 2], expected input with shape [*, 3, 2], but got input of size[2, 6]
      tpu: ok shape=(2, 6) sum=-0.0000
    layer_norm_module
      cpu: RuntimeError: Given normalized_shape=[768], expected input with shape [*, 768], but got input of size[2, 512]
      tpu: RuntimeError: layer_norm(): failed to broadcast tensor: <unknown>:0: error: Cannot broadcast input: 'tensor<768xf32>' to target shape tensor<2x512>
    ```

  * The failed cases also log `[SetAsError] Failed to set materialization
    error: FAILED_PRECONDITION` twice each (noise).
* Confirmed facts:
  * `csrc/ops/layer_norm/layer_norm_aten_kernels.cc` `ValidateInputs`
    (upstream/main) checks only: floating dtype, non-empty
    `normalized_shape`, and `normalized_shape.size() <= input.dim()`.
    It does not check that the input ends with `normalized_shape`, and it
    does not check weight and bias shapes.
  * No commit in `afd6943a..upstream/main` touches `csrc/ops/layer_norm`.
  * `rms_norm` and `group_norm` on TPU already raise good errors
    (precedent, S3-E3).
* Hypotheses: the backward kernel (`AtenLayerNormBackward`) needs the
  same checks. Under `torch.compile` the meta function catches these
  shapes already. The noisy `[SetAsError]` log goes away when the
  error is raised before dispatch.
* Proposed scope: in `ValidateInputs`, check that the trailing input dims
  equal `normalized_shape`, and that weight and bias (if given) have
  shape `normalized_shape`. Use messages close to ATen. Call it from the
  forward and backward kernels. Add error tests. Do not touch the
  lowering, the dtype rules (open PR 5691), or performance.
* Validation plan:
  * Error tests in `tests/errors_test.py` per the `tt-write-error-tests`
    skill: the 4 cases from the repro (shape, same numel, module, weight).
  * Build a wheel on `afd6943a` in the worker worktree (BUILD_NOTES.md),
    install it in the worktree venv, run the tests under the TPU lock:
    fail before, pass after. Run the existing layer_norm tests in
    `tests/ops_unit_test.py`.
  * `BASE_SHA=upstream/main nox -s lint`.
* Project benefit: removes a silent wrong result and an unclear MLIR
  message for a very common layer.
* Issue first: no, it is a small validation fix (the same as recent
  merged "[Ops][x] Validate ..." PRs). An issue is fine if the reviewer
  wants one.
* Overlap: none. Searched "layer_norm normalized_shape", "layer_norm
  validate", "layer norm shape". PR 5691 (open) changes the bf16 weight
  dtype rule in the same file. It is a different topic, but a merge
  conflict is possible. Keep the change in `ValidateInputs`.
* Finish estimate: C++. Wheel build 20-40 min with the cache. About
  2-2.5 h in total.

## 3. Return an empty result when indexing an empty dimension

* Rank 3. `x[idx]` and `index_select` on a tensor with a size-0 indexed
  dim abort the whole process (SIGABRT, LLVM fatal error). Found in a
  RetinaNet training step (S3-E2).
* Evidence (my re-run, `scratch/merge-s3/empty.log`):
  * Command: `cd $COORD/scratch/S3-E2 && flock /root/tpu.lock bash
    run_empty.sh`
  * Output (one case, the other 3 abort the same way):

    ```text
    == float_empty_idx
    tensor([])
    loc("index/index"): error: slice size (1) is out of bounds for operand dimension (0) at index 0
    "stablehlo.gather"(...) {} : (tensor<0xf32>, tensor<0x1xi64>) -> ( ??? )
    *** SIGABRT received by PID 180977 ...
    ```

  * Abort cases: int64 and f32 `x[i]` with `x` of shape `[0]`, `x` of
    shape `[0, 4]`, and `torch.index_select(x, 0, i)`.
    No abort: non-empty `x` with an empty index, and a bool mask then
    index.
  * The print shows the right empty result first. The abort comes later,
    from the background compile of the lazy graph.
* Confirmed facts:
  * `csrc/ops/index/index_aten_kernels.cc` `BuildIndexShlo` (upstream/main)
    sets `gather_slice_sizes` to 1 for each indexed dim. If that dim has
    size 0, `stablehlo::Gather` type inference fails, and MLIR calls
    `llvm::report_fatal_error`.
  * `index_select` uses the same builder.
  * The only commit in `afd6943a..upstream/main` on `csrc/ops/index` is a
    BUILD file change (c98dcfd1f).
* Hypotheses: `gather`, `take`, and `embedding` with an empty weight may
  have the same problem. Check them and fix them only if they share the
  same code path.
* Proposed scope: when an indexed operand dim has size 0 (then the index
  must be empty), skip the gather and return an empty tensor of the
  output shape. Cover `index.Tensor` and `index_select`. Add tests. Do
  not change the non-empty path.
* Validation plan:
  * Tests in `tests/ops_unit_test.py`: the 4 abort cases, compared with
    CPU, each run with `.cpu()` to force execution. Run each test in a
    subprocess while the bug exists, or just confirm failure before the
    fix by the repro script (an abort kills the test runner).
  * Build a wheel on `afd6943a`, run under the TPU lock.
  * `BASE_SHA=upstream/main nox -s lint`.
* Project benefit: removes a process abort (Jupyter kernel dies, no
  traceback) in detection losses and in any code that filters to an
  empty tensor and then indexes it.
* Issue first: no, a small crash fix. Optional short issue.
* Overlap: none. Searched "empty index", "index_select empty", "slice
  size out of bounds", "Failed to infer result type". PR 6490 (merged)
  validates gather index dtype only.
* Finish estimate: C++. About 2-2.5 h with the build. A small risk: the
  dynamic-shape path of `BuildIndexShlo` may need care.

## Other findings, ranked (not chosen now)

4. `max_pool1d/2d/3d` and `avg_pool1d` abort when the output size is < 1
   (S3-E3 F1; `nn.MaxPool2d(3)` on a 2x2 map). C++ 30-40 lines in
   `csrc/ops/pooling/pooling.cc` `GetPoolingOutputSize`. Strong, but the
   `std::max(out, 1)` clamp came from a refactor (5fc41fd4d), and
   `ceil_mode` needs care. Good next task after the three above.
5. SDPA with mismatched q/k/v shapes aborts (LLVM ERROR) or gives MLIR
   text (S3-E3 F2). C++ about 40 lines of shape checks. Strong, but SDPA
   has many in-flight bot PRs (routing, philox), so conflict risk is
   higher.
6. DataLoader workers take the TPU because `torch.manual_seed` initializes
   it (S3-E4 F1). High impact (`num_workers>0` before the first TPU op
   crashes; otherwise error spam). But the code has an internal TODO
   (b/572861883) for exactly this, and the fix is a design change in
   seeding. Best as a well-written issue, not a PR from us.
7. Some TPU kernels drop channels_last strides; breaks `view()` and the
   compiled diffusers VAE backward (S3-E2 F1). High impact, but the right
   fix (central or per kernel) needs a maintainer decision. Issue first.
8. `uniform_` / `exponential_` accept invalid parameters (S3-E3 F4). C++
   about 15 lines. Easy, lower impact.
9. `GradScaler` ops missing (S3-E1 F2). An earlier PR 2464 was closed
   unmerged; ask maintainers first. 100-150 lines C++.
10. `TORCH_TPU_ENABLE_DEBUG_CHECKS=1` checks indices only for `take` and
    `histc` (S3-E3 F5). Series of small PRs; ask in an issue first.
11. Feature gaps with clear errors: distributions samplers (gamma,
    dirichlet, poisson, binomial), `torchvision::nms`/`roi_align`,
    `linalg_matrix_exp`. Too large for one session.
12. Low: profiler host-event noise (S3-E4 F2), `torch.Stream("tpu")`
    without index (still broken on main), dynamism docs wording.
