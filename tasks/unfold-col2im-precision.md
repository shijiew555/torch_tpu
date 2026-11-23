# unfold-col2im-precision (worker 2)

## Session 3 refresh (in progress)

* Test branch unfold-col2im-precision-s3 = backup commits on afd6943a.
  Worktree .claude/worktrees/unfold-col2im-precision, venv with torch
  2.16.0.dev20261112+cpu, libtpu 0.0.53.1, afd6943a wheel + Python overlay.
* Before (afd6943a wheel), repro: F.fold 0.0156, F.unfold bwd 0.0114,
  Tensor.unfold bwd 0.00717, same with matmul precision highest;
  torch.compile F.fold 0.0156. ops_unit_test '*col2im* *im2col* *unfold*
  *fold*': 15 run, 3 new tests FAIL (1.56e-2, 1.29e-2, 5.56e-3).
* Wheel build of -s3 running.

## Session 2

State: done, validated. PR draft ready. Nothing posted.

* Branches (pushed to origin): unfold-col2im-precision 0cf8dcc1,
  unfold-col2im-precision-pr 3de2bd7f (one squashed commit, same tree).
* Base: upstream/main 3c6acc43.
* Fix: f32 HIGHEST (input) / DEFAULT (0/1 weights) precision config on the
  convolution in unfold backward and col2im, with EXPLICIT_PRECISION_OK markers.
* Tests: 3 new f32 precision tests in tests/ops_unit_test.py; test_col2im_fold
  now uses default tolerances.

## Results (TPU v5e, //tests:ops_unit_test_tpu-v5lite, --//:test_wheel=True)

* Before (fix reverted, same tests): 3 new tests FAIL. Max abs err:
  fold 1.56e-2, im2col bwd 1.29e-2, unfold bwd 5.56e-3 (allowed 1e-5).
  test_col2im_fold passes at default tolerances even before (arange input).
  Log: tmp/unfold-col2im-precision/test_before.log.
* After: filter '*col2im* *im2col* *unfold*': 15 tests OK, target PASSED.
  Log: tmp/unfold-col2im-precision/test_after.log.
* Build: 1510 s incremental (13,549 disk cache hits).
* nox -s format: no changes. BASE_SHA=upstream/main nox -s lint: pass.
* Not done: performance/compile-time measurement of HIGHEST.
