# ckpt-tpu-rng-state (session 3 worker)

State: done. PR branch ckpt-tpu-rng-state-pr aae1baaf (squashed on upstream/main 82d0a494), draft in pr_desc/ckpt-tpu-rng-state.md.

* Base: upstream/main 82d0a494. Worktree: .claude/worktrees/ckpt-tpu-rng-state.
* Backup branch: ckpt-tpu-rng-state 58c53902 (pushed).
* Fix: metaclass property `_DeviceModuleMeta._initialized` (bad-fork check +
  backend `_is_initialized()`, no warning). `is_initialized()` returns it.
  torch.tpu reaches it through `private_fallback`.
* Only torch/utils/checkpoint.py reads `_initialized` in torch 2.16 nightly.
* Tests: `RngCudaRefTest.test_eager_activation_checkpointing_dropout`
  (reentrant, non_reentrant) in tests/rng_cuda_ref_test.py; one assert in
  tests/device/device_module_testing.py `test_initialized`.
* Validation: venv (afd6943a wheel) + overlays (upstream/main Python, and
  afd6943a Python + patch). Scripts and logs: scratch/ckpt-tpu-rng-state/.

| Check | Before | After |
| --- | --- | --- |
| repro ckpt_dropout.py, TPU grad diff | 1.51 (both) | 0.0 (both) |
| TransformerEncoderLayer AC rel grad diff | 0.705 | 0 |
| new rng_cuda_ref_test cases (2) | fail | pass |
| rng_cuda_ref_test full, afd overlay | - | 70 OK (1 skip) |
| rng_test full, afd overlay | - | 21 OK |
| device_module_tpu_test (41) | test_initialized fails | OK |
| public_api_test, annotations_test | - | OK |
| nox -s lint (BASE_SHA=upstream/main) | - | pass |

Env limits: on the upstream/main overlay, 11 compiled RNG tests fail before
and after (wheel lacks `get_non_default_device_layout`). tt_apis_test needs
torch 2.14. device_module_no_tpu_test needs a non-TPU host.
* PR branch: new tests + test_initialized pass on its overlay; lint passes.
