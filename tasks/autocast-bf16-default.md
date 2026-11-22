# autocast-bf16-default

State (session 3 refresh, in progress): PR branch rebuilt on upstream/main
82d0a494. Final checks running.

* Branches on origin:
  * autocast-bf16-default: backup, now 3 commits on e411f09d. New commit
    adds the PyTorch test_autocast exclusions (see below).
  * autocast-bf16-default-pr: old clean branch on e411f09d. Superseded.
  * autocast-bf16-default-pr2: new clean branch, 1 commit on 82d0a494.
* Env: worktree venv, torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1,
  afd6943a wheel + Python source overlay of the tested commit. TPU v5e.
  Scripts and logs: .claude/tmp/autocast-bf16-default-s3/.
* New finding: with the fix, PyTorch test_autocast (TorchTPU runner,
  PyTorch tests at 82b208e2) fails 2 tests: TestAutocastDevice
  test_cache_disabled and test_cast_cache_is_global. They count casts to
  float16 (hard-coded). Fix: exclude both in tests/torch/torch_test_configs.py.
  Upstream PyTorch could use torch.get_autocast_dtype(device) instead.
* Results (before = base, after = fix):
  * Repro: float16 / inf -> bfloat16 / 360448.0 (installed wheel, e411,
    main 82d0a494 all reproduce the bug).
  * amp_test: new 3 tests pass after. test_autocast_works now passes.
    test_autocast_rnn errors before and after on main (needs C++ 5bf70a27,
    not in afd6943a binaries).
  * device_module_tpu_test: 41 OK before and after.
  * PyTorch test_autocast: 9 OK (2 skipped) before; after: 2 fail without
    exclusion, pass with it.
  * nox -s lint (BASE_SHA=upstream/main): pass. nox -s format: no changes.
* Pre-existing (not this task): PyTorch test_ops test_fake_autocast aborts
  the process at index_reduce (scatter.cc:412 CHECK, rank 2 vs 1), before
  and after the fix.
* PR + issue draft: coord/pr_desc/autocast-bf16-default.md
