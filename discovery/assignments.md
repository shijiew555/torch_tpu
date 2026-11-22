# Explorer assignments - round 1

Read `discovery/context.md` first. It lists claimed issues, open PRs, and
fixed topics. Do not report anything on those lists.

## Rules for all explorers

* Goal: find bugs, errors, and crashes that block common torch_tpu use.
  Find them by running trials. Reproduce each one.
* Python: system `python` (torch_tpu 0.3.0, torch 2.14.1+cpu). Install extra
  packages (for example transformers) with pip only if needed; do not change
  the torch or torch_tpu versions.
* TPU work: `flock /root/tpu.lock <command>`. One TPU step per lock. Do not
  hold the lock while you read code. Set `timeout` on every TPU run (a hang
  is also a finding).
* Scratch: `/root/torch_tpu/.claude/tmp/<ID>/`. Output:
  `/root/torch_tpu/.claude/coord/discovery/<ID>.md`. Do not run git in coord.
* Compare each TPU result with CPU (same inputs, sensible tolerance).
* For each finding, check it is not fixed on upstream/main: read the source
  with `git show upstream/main:<path>`, `git log upstream/main -S<text>`, and
  the open PR list. Search issues:
  `gh issue list -R google-pytorch/torch_tpu --state all --search "<words>"`.
* Find the root cause location in the source (file and function). Say if the
  fix is Python-only (cheap) or C++ (needs a build).
* Time box: about 75 minutes of work. Stop at T0+6h (08:32:14Z) at latest.
* Report format per finding: title; minimal repro script path; command and
  actual output; expected output; affected versions (0.3.0, main by
  source reading); root cause hypothesis with file:function; Python or C++
  fix; overlap check result; your impact rating (high/medium/low) with one
  reason. Separate confirmed facts from hypotheses. Report at most 5
  findings, best first. Write ASD-STE100 style. No em dash.
* Also list "tried, works" areas in one short list, so later rounds skip them.

## E1 - Hugging Face model training and inference (TPU)

Area: the most common user path. Small randomly initialized models from
`transformers` (pip install allowed; do not download large
weights; build models from config).
Questions:
* Do Llama, GPT-2, BERT, T5 (or Qwen/Gemma if present) configs run forward,
  backward, and an AdamW step on TPU in eager mode? With `torch.compile`?
* Does `model.generate()` work (greedy, sampling, with KV cache, with
  `attention_mask` padding)? Any crash, hang, recompilation storm, or wrong
  tokens versus CPU?
* bf16 (`torch_dtype=torch.bfloat16`) and `torch.autocast("tpu")`?
* `Trainer` or `accelerate` on device "tpu": does device detection work?
Evidence: script, loss/logit diff versus CPU, error traceback, compile count.

## E2 - torch.compile common patterns (TPU)

Area: `torch.compile(backend="tpu")` and the default backend on TPU tensors.
The compiler backend is mostly Python in `src/torch_tpu/_internal/`.
Questions:
* Graph breaks (print, `.item()`, data-dependent control flow), in-place
  mutation of inputs, buffers mutated in forward, `torch.no_grad` and
  `inference_mode`, `requires_grad` changes, returning views or aliases of
  inputs, outputs that alias each other.
* `dynamic=True` and `mark_dynamic`: does it run and give right results?
* Autograd through compiled modules: higher-order grads, `retain_graph`,
  `detach`, custom `autograd.Function`, `torch.utils.checkpoint` inside a
  compiled region.
* `fullgraph=True` errors: are messages clear? `torch._dynamo.reset()` and
  recompile limits.
Avoid: LSTM/GRU (#6677), storage-offset recompiles (#5000), precision graph
break (#4410), compile cache precision key (#6393).
Evidence: script, eager vs compiled vs CPU outputs, traceback, recompile count
(`TORCH_LOGS=recompiles`).

## E3 - Core op correctness on common model ops (TPU)

Area: ops used by common models, run against CPU.
Questions:
* Use `torch.testing._internal.common_methods_invocations.op_db` sample
  inputs for a chosen list of common ops (indexing, gather/scatter,
  index_put, masked_fill/where, cumsum/cumprod, sort/topk/argsort,
  embedding, losses, normalization, conv1d/2d, pooling, interpolate, matmul
  variants, reductions, type promotion, bf16/fp16/int dtypes). Use
  non-contiguous inputs, empty tensors, 0-d tensors, odd shapes.
* Which ops give wrong values, crash (abort the process), or raise "not
  implemented"? Run each op in a subprocess so an abort does not stop the
  sweep.
* Check autograd (`gradcheck`-style compare against CPU grads) for the
  common ones.
Avoid ops on the claimed or fixed lists (index_add_, median, gcd, mish,
logaddexp, huber_loss, multi_margin_loss, max_unpool2d, var/std dyn,
group_norm dtype, round decimals, hardshrink, softshrink, view_as_complex).
Evidence: table of op, dtype, failure kind, max error; repro per finding.

## E4 - Data movement, device semantics, and runtime APIs (TPU)

Area: moving data and managing the device.
Questions:
* `.to("tpu")`, `.to(dtype)`, `non_blocking=True`, `pin_memory`, `copy_`
  across devices and dtypes, `.cpu()` of views and non-contiguous tensors,
  `tensor.tolist()`, `.item()`, `torch.tensor(..., device="tpu")`,
  `torch.from_numpy` then `.to`.
* `torch.save` / `torch.load(map_location=...)` of TPU tensors and state
  dicts, `load_state_dict` with `assign=True`, `torch.device("tpu:0")`
  handling, meta-device init then `to_empty(device="tpu")`.
* RNG: `torch.manual_seed`, `torch.tpu.manual_seed`, `get_rng_state` /
  `set_rng_state`, `torch.Generator(device="tpu")`, reproducibility of
  dropout and randn.
* `torch.tpu.*` and `torch.accelerator.*` APIs: synchronize, memory stats,
  empty_cache, streams/events, `torch.get_device_module`.
* DataLoader with `num_workers>0` and `pin_memory=True`.
Avoid: strided view write/read abort (#6675), stream API FR (#6216),
memory APIs papercut (#4413), memory stats before init (PR 6738).
Evidence: script, actual vs expected, traceback.

## E5 - Training loop components and single-chip distributed (TPU)

Area: pieces every training loop uses.
Questions:
* Optimizers: SGD (momentum, nesterov), Adam/AdamW (`foreach`, `fused`,
  `amsgrad`, `capturable`), Adagrad, RMSprop; LR schedulers; compare a few
  steps with CPU. `clip_grad_norm_` / `clip_grad_value_` (foreach).
* `torch.amp.GradScaler("tpu")` and autocast with fp16/bf16.
* `torch.utils.checkpoint` (reentrant and non-reentrant), `torch.func`
  (`vmap`, `grad`, `jacrev`), `nn.utils.parametrizations` (weight_norm,
  orthogonal, spectral_norm), `torch.distributions` sampling and log_prob.
* Single-chip distributed: `init_process_group` with world_size=1 on the TPU
  backend, DDP wrap, FSDP2 `fully_shard` on a 1-device mesh, DTensor ops,
  functional collectives, `torch.distributed.checkpoint` save/load.
Avoid: FSDP2 `_chunk_cat` (#6729), DDP+compile graph breaks (#4411),
barrier (PR 6744), BatchNorm running stats (#5667).
Evidence: script, param diff vs CPU after N steps, traceback.

## E6 - Build and test feasibility (CPU, no TPU lock needed)

Area: how a fix can be built and tested on this machine. Every candidate
depends on this. Do not fix #6508 (assigned).
Questions:
* Install bazelisk (the version in `.bazelversion`) and nox with pip or a
  download. Does `nox -s lint` run on a clean upstream/main worktree?
* Make a worktree of upstream/main under `/root/torch_tpu/.claude/tmp/E6/`
  (use `git worktree add`, do not touch the main checkout). Try
  `bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel
  --config=no_rbe` with the #6508 workaround
  (`--spawn_strategy=sandboxed,local`). Run it in the background with a log.
  Measure progress rate and estimate total time. Is there a remote cache
  config usable without credentials (for example
  `helion_public_caching_readwrite` or a read-only public cache in
  `.bazelrc`)? Report what speeds it up.
* Can one Python test target (for example a test in `tests/` that has a
  CPU config) be run with `bazel test` faster than a full wheel?
* For Python-only fixes: can a changed file from `src/torch_tpu/_internal/`
  be overlaid on the installed 0.3.0 package and tested (copy to a scratch
  location; do not edit the installed files in place while other explorers
  run)? How different is 0.3.0 Python code from upstream/main (`diff` the installed files with
  `git show upstream/main:src/torch_tpu/_internal/...`)?
* Time box: 2.5 hours. Leave the build running in the background only if the
  controller agrees; otherwise stop it and record the log path and rate.
Evidence: commands, durations, disk use, failure logs, a recommended
build-and-test recipe for the fix phase.

## Scheduling advice for the controller

* Run E6 first, paired with E1 (E6 uses CPU only, so it does not compete
  for the TPU lock).
* Then E2 + E3, then E4 + E5. Each TPU pair shares the lock; that is fine,
  their steps are short.

# Session 3 - explorer assignments

Plan written 2026-11-22T22:50Z by the explorer controller.
T0 = 2026-11-22T22:40:03Z. Wrap-up T0+6h = 2026-11-23T04:40:03Z.
Hard stop T0+7h = 2026-11-23T05:40:03Z.

## Suggested launch order (at most two at once)

* Wave A: S3-E1 (TPU, many short runs) + S3-E3 (TPU-light sweep).
* Wave B: S3-E2 (TPU) + S3-E4 (TPU-light).
* Time box per explorer: about 90 minutes of work.
  Stop new work at T0+6h in any case.

## Rules for all Session 3 explorers

* Read first: `discovery/context.md` (claimed topics, may be stale), the
  "Considered but not chosen" list in `discovery/candidates.md`, and the
  "Tried, works" and "Dead ends" lists in `discovery/E1.md` to `E6.md`.
  Do not report those topics again unless you show something new.
* Do not repeat the five earlier tasks: autocast bf16 default,
  unfold/col2im precision, compiled output aliases, `unique(sorted=False)`,
  pin_memory DeprecationWarning (`tasks/`, `pr_desc/`).
* Python and TPU: use the discovery venv
  `/root/torch_tpu/.claude/worktrees/build-base/.venv/bin/python`
  (Python 3.14, torch 2.16.0.dev20261112+cpu, torch_tpu wheel of
  `afd6943a`, libtpu 0.0.53.1). Do not use the system Python. Do not
  change torch or torch_tpu in this venv. If you need an extra package
  (transformers, torchvision, diffusers, timm, expecttest), install it
  with `uv pip install --python <venv python> <pkg>` and say so in your
  report. For torchvision, pick the nightly that matches torch
  (`--index-url https://download.pytorch.org/whl/nightly/cpu --no-deps`).
* TPU work: `flock /root/tpu.lock timeout <s> <command>`. One TPU step per
  lock. Do not hold the lock while you read code or wait. Set a timeout
  on every TPU run (a hang is also a finding). Use
  `faulthandler.dump_traceback_later` for hang stacks.
  CPU-only `torch.compile` scripts also open the TPU (E2 dead end), so run
  them under the lock too. Build op lists with
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` to avoid opening the TPU.
* Scratch: `/root/torch_tpu/.claude/tmp/<ID>/`. Copy the repro scripts
  that matter to `$COORD/scratch/<ID>/`. Report:
  `$COORD/discovery/<ID>.md`. Do not run git in `$COORD`; the explorer
  controller commits your files.
* Compare each TPU result with CPU (same inputs, sensible tolerance).
* Check every finding against current upstream/main (`82d0a494`, 146
  commits after `afd6943a`): read the source with
  `git -C /root/torch_tpu show upstream/main:<path>`, run
  `git -C /root/torch_tpu log afd6943a..upstream/main --oneline -S<text>`
  and look at the open PR list. Search issues and PRs, all states:
  `gh issue list -R google-pytorch/torch_tpu --state all --search "<words>"`
  and `gh pr list -R google-pytorch/torch_tpu --state all --search
  "<words>"`. A finding that is fixed on main, assigned, claimed, or
  linked to an open PR is a dead end. Say so in one line.
* Find the root cause location (file and function). Say if the fix is
  Python-only or C++ (needs a rebuild), and give a size estimate.
* Prefer small fixes (under about 50 lines plus tests). A new kernel is
  fine if it is small and has a clear precedent in the repo.
* Report format per finding: title; repro script path; command and actual
  output; expected output; status on main (by source reading); root cause
  with file:function; Python or C++, size; overlap check result; impact
  (high/medium/low) with one reason. Separate confirmed facts from
  hypotheses. At most 5 findings, best first. Then one short "Tried,
  works" list and one short "Dead ends" list. ASD-STE100 style. No em
  dash. Lead with the result.

## S3-E1 - Training loop components and single-chip distributed (TPU)

Area: pieces that every training loop uses. This was E5 in round 1; it
never ran. HF model training itself works (E1), so focus on the
components and options around it.
Questions:
* Optimizers on TPU vs CPU after 3-5 steps: SGD (momentum, nesterov,
  weight_decay), Adam/AdamW (`foreach=True/False`, `fused=True`,
  `amsgrad`, `capturable`, `maximize`), Adagrad, RMSprop, Adafactor,
  NAdam, RAdam, LBFGS. bf16 params. `optimizer.state_dict()` save, load
  with `map_location`, resume. `zero_grad(set_to_none=False)`.
* LR schedulers (OneCycle, CosineAnnealing, LinearLR, SequentialLR,
  ReduceLROnPlateau with a TPU loss tensor).
* `clip_grad_norm_` (foreach on/off, `error_if_nonfinite`, inf/nan
  grads), `clip_grad_value_`, `get_total_norm`.
* `torch.amp.GradScaler("tpu")`: scale, unscale_, step with inf grads
  (step must skip), update. Does `GradScaler()` with default device work
  for a TPU model? Autocast fp16 training with GradScaler.
* `torch.utils.checkpoint` (reentrant and non-reentrant, with RNG
  preservation for dropout), `torch.func` (`vmap`, `grad`, `jacrev`,
  `functional_call`), `nn.utils.parametrizations` (weight_norm,
  orthogonal, spectral_norm), `torch.distributions` sample/rsample and
  log_prob (Normal, Categorical, Bernoulli, Beta, Dirichlet, Gamma).
* `torch.optim.swa_utils.AveragedModel`, EMA updates with
  `torch._foreach_lerp_`/`_foreach_mul_`.
* Single chip distributed: `init_process_group` with world_size=1 on the
  TPU backend, DDP wrap and a few steps, FSDP2 `fully_shard` on a
  1-device mesh, DTensor basic ops, functional collectives,
  `torch.distributed.checkpoint` save and load.
Avoid: FSDP2 `_chunk_cat` (#6729, PR 6728), DDP+compile graph breaks
(#4411), barrier (fixed 93558d35), BatchNorm running stats (fixed),
`_fused_dropout` (PR 6665), `F.linear` grad buffer (#6593 closed),
conv/norm backward memory (#6882).
Evidence: script, param diff vs CPU after N steps, traceback.
Label: TPU (many short runs; release the lock between runs).

## S3-E2 - Vision and diffusion models (TPU)

Area: the second most common user path after HF text models. Round 1
tested only ResNet, ConvNeXt, ViT, CLIP from transformers.
Questions:
* torchvision models, random init, small inputs: forward, backward, one
  SGD step, eager and `torch.compile(backend="tpu")`, vs CPU. Cover
  ResNet, MobileNetV3, EfficientNet, RegNet, Swin, ConvNeXt, DenseNet,
  Inception/GoogLeNet (aux outputs), ShuffleNet (channel shuffle),
  segmentation (DeepLabV3, FCN), detection (Faster R-CNN, RetinaNet,
  SSD, Mask R-CNN) and video (r3d_18) if they run in reasonable time.
* `torchvision.ops`: `nms`, `batched_nms`, `roi_align`, `roi_pool`,
  `box_iou`, `sigmoid_focal_loss`, `deform_conv2d`. Does a missing
  torchvision kernel give a clear error, a CPU fallback, or a crash?
* torchvision.transforms.v2 on TPU tensors (resize, normalize, crop,
  flip, color jitter) vs CPU.
* diffusers: a small UNet2DConditionModel and AutoencoderKL from config,
  a DDPM/DDIM scheduler step loop, forward and backward, eager and
  compiled, bf16. A tiny DiT or SD3-style transformer if available.
* Missing ops these models hit (for example `adaptive_max_pool2d`
  backward, `upsample_trilinear3d`, `kthvalue`; all three are still in
  `UNREGISTERED_OPS` on upstream/main). For each missing op, estimate how
  many of the tested models need it, and the kernel size with a precedent
  in `csrc/ops/`.
* GroupNorm, channels_last, `F.interpolate` with `scale_factor` and
  `align_corners`, `F.grid_sample`, `pixel_unshuffle`, `conv3d`,
  `conv_transpose3d`, `max_pool3d`, `avg_pool3d`.
Avoid: group_norm bf16/f32 dtype (#6015), conv default precision
mismatches (E1 dead end; compare with
`torch.set_float32_matmul_precision("highest")` before you report a
numeric diff).
Evidence: script, output/grad diff vs CPU, traceback, compile time.
Label: TPU.

## S3-E3 - Error path parity sweep (TPU-light)

Area: user-facing errors. Many recent merged PRs on main add input
validation ("[Ops][topk] Validate k", "[Ops][eye] Validate ...",
"[Ops][roll]", "[Ops][flip]", "[Ops][softmax] Validate dim"). Bad input
that aborts the process, gives an internal error, or silently returns
garbage on TPU (where CPU raises a clear error) is a real bug and is
very likely to be merged.
Method:
* Use `op_db` `error_inputs` (and `error_inputs_sparse` = skip) from
  `torch.testing._internal.common_methods_invocations` for common ops.
  Run each op's error inputs on CPU and on TPU. Run each op in a
  subprocess (or a chunk of ops per subprocess with a crash handler) so
  an abort does not stop the sweep. Use one lock per chunk.
* Classify on TPU: (a) process abort or segfault, (b) a hang, (c) no
  error where CPU raises (and what is returned), (d) error with an
  internal or unclear message (C++ file paths, "INTERNAL", "Check
  failed", MLIR verifier text, "unknown error"), (e) a different
  exception type than CPU (`RuntimeError` vs `IndexError`/`ValueError`).
  Lazy execution can delay an error; force it with `.cpu()` and record
  where it surfaces.
* Also try common user mistakes outside op_db: mixed CPU and TPU tensors
  in binary ops, `index_select`/`gather`/`embedding` with out-of-range
  indices (does it raise, clamp, or return garbage? CPU raises), `cat`
  with mismatched dtypes or shapes, `view` on non-contiguous, wrong
  `out=` dtype or device, matmul shape mismatch, `cross_entropy` with
  target out of range or wrong dtype.
* Read the `tt-write-error-tests` skill
  (`.agents/skills/tt-write-error-tests/SKILL.md`) and
  `tests/errors_test.py` on upstream/main to see how errors are tested
  and how GPU parity is judged (out-of-range indices on GPU raise
  device-side asserts; say how TPU should behave per the skill).
Avoid ops validated on main since `afd6943a` (`git log
afd6943a..upstream/main --oneline | rg -i "validat|reject"`), and topk
(#6865, PR 6866), eye (#6863, PR 6864), masked_select.out (PR 6838),
linalg.solve_ex (PR 6918), and anything in the open PR list.
Evidence: table of op, input, CPU behavior, TPU behavior, class (a-e);
repro per finding; check that each finding still reproduces by source
reading on upstream/main.
Label: TPU-light (short per-op runs, chunked under the lock).

## S3-E4 - Autograd features, tooling, and runtime extras (TPU-light)

Area: features that real training and debugging code uses, outside of
plain ops. None of these were tested in round 1.
Questions:
* Autograd features on TPU: `register_hook` on tensors, module
  `register_full_backward_hook` and `register_forward_pre_hook`,
  `torch.autograd.graph.saved_tensors_hooks` and
  `torch.autograd.graph.save_on_cpu(pin_memory=True)` (activation
  offload; common for memory saving), `set_detect_anomaly(True)`,
  `torch.autograd.grad` with `allow_unused`, `is_grads_batched`,
  forward-mode AD (`torch.autograd.forward_ad`), `inference_mode` tensors
  used later, `retain_grad` on non-leaf, `requires_grad_` on views.
* `torch.use_deterministic_algorithms(True)` on TPU (index_put,
  scatter_add, cumsum): any error or warning that CPU/CUDA would not
  give? `torch.backends.*` flags that HF and timm read.
* Profiler: `torch.profiler.profile(activities=[ProfilerActivity.CPU,
  ProfilerActivity.PrivateUse1])` around a TPU training step,
  `key_averages()`, `export_chrome_trace`, `record_function`, schedule
  with `tensorboard_trace_handler`. Crash, empty TPU events, or noise?
* Memory tools: `torch.accelerator.memory_allocated/max_memory_allocated/
  reset_peak_memory_stats` across a step; do numbers make sense?
  (Avoid #4413 papercuts and PR 6738.)
* Multiprocessing: `torch.multiprocessing` spawn of a worker that uses
  TPU tensors; sending a TPU tensor through a `mp.Queue` (should raise
  clearly or work); DataLoader with `persistent_workers=True`,
  `pin_memory_device="tpu"`, `prefetch_factor`.
* Misc: `torch.Tensor.__repr__`/`print` of large, bf16, complex, empty,
  and non-contiguous TPU tensors; `torch.testing.assert_close` with TPU
  tensors; `tensor.numpy()` error message; `torch.set_printoptions`;
  sparse COO tensors `.to("tpu")` (clear error?); `torch.nested`.
* Re-check round 1 "Considered but not chosen" items on upstream/main by
  source reading: `torch.Stream("tpu")` without index (PR 6737 landed as
  e9035bbb; is the bug gone?), the docs wording on default dynamism
  (E1 F4). One line each.
Evidence: script, actual vs expected, traceback.
Label: TPU-light.
