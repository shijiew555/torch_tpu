# unfold-col2im-precision (worker 2)

## Session 3 refresh (done, 2026-11-23)

State: done. Ready for user approval. Nothing posted.

* Branches on origin:
  * unfold-col2im-precision-pr2 = be7d0e6d: one squashed commit on
    upstream/main 82d0a494. Same diff as the tested branch. Use this.
  * unfold-col2im-precision-s3 = 50615db4: backup commits cherry-picked
    on afd6943a + docstring tweak (review). Tested branch.
  * unfold-col2im-precision / -pr: session 2, superseded, unchanged.
* Env: worktree venv (.claude/worktrees/unfold-col2im-precision/.venv),
  torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1, Python 3.14, TPU v5e.
  Before = afd6943a wheel; after = wheel built from -s3 (1153 s, 13,603
  disk cache hits). Python overlay of worktree source (test_utils is not
  in the wheel). Scripts and logs: scratch/unfold-col2im-precision/.
* Repro (max abs err vs CPU), before -> after:
  F.fold 1.56e-2 -> 4.8e-7; F.unfold bwd 1.14e-2 -> 4.8e-7;
  Tensor.unfold bwd 7.2e-3 -> 0. matmul precision "highest" does not
  change the before values. torch.compile(backend="tpu") F.fold
  1.56e-2 -> 4.8e-7.
* ops_unit_test '*col2im* *im2col* *unfold* *fold*' (15 tests): before 3
  new tests FAIL (1.56e-2, 1.29e-2, 5.56e-3); after OK (also at 50615db4).
* ops_test '*unfold*' (OpInfo vs pinned GPU goldens, downloaded from
  tests/goldens.pin, sha256 checked), compute_grad False and True: 2 OK
  each, before and after.
* pr2: BASE_SHA=upstream/main nox -s lint pass; nox -s format no change.
* Review: precedent now also in scatter.cc (same pattern); commit message
  says so. Test docstrings reworded. No other findings.
* Benchmark (coordinator follow-up): paired ABAB x3 under one TPU lock,
  extracted before/after wheels via PYTHONPATH, same venv, median per
  call with torch.accelerator.synchronize. LARGE COST for f32:
  fold/unfold ViT 224 p16 (batch 8) 92 -> 310 ms (3.4x), 96 -> 314 ms
  (3.3x); 64ch 56x56 k3: fold 6.1 -> 11.2 ms (1.8x), unfold fwd+bwd
  7.1 -> 12.1 ms (1.7x); Tensor.unfold bwd 16x16000 w400 h160 13.8 ->
  24.8 ms (1.8x). First call also slower (867 -> 1168 ms ViT fold).
  bf16 control: 1.00x everywhere. Rounds agree within 0.01x.
  Note: even "before" is slow (92 ms for a 1.2M-element fold), so the
  convolution lowering itself is costly. An exact gather/pad-and-add
  lowering might be both exact and faster (untested idea).
  Script/log: scratch/unfold-col2im-precision/bench.{py,sh,jsonl}.
  Fix not changed; waiting for coordinator decision.

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
