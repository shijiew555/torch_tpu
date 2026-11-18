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
