# autocast-bf16-default (worker 1)

State: done, 06:31Z. Waiting for user approval of issue + PR.

* Branches (pushed to origin): autocast-bf16-default 5d6f9787 (backup),
  autocast-bf16-default-pr 50a7b83a (clean, 1 commit on upstream/main e411f09d).
* Fix: _loader._init_device_impl calls torch.set_autocast_dtype("tpu", bf16).
  Tests in amp_test.py (3 new); device_module_testing setUp default fp16 -> bf16.
  Docs note on thread scope in docs/api_reference.md.
* Results (overlay on 0.3.0 binaries, TPU v5e, logs in .claude/tmp/autocast-bf16-default/):
  * amp_test new tests: fail before (3), pass after.
  * amp_test test_autocast_works: fails before and after (identical). Cause: 0.3.0
    binaries predate 2b69edb1 (linear f32 accumulation lowering). Env artifact.
  * device_module_tpu_test: 41 OK before and after.
  * nox -s lint: pass (backup and PR branch). nox -s format: no changes.
* Coverage: main thread yes; autograd backward thread yes; new Python thread no
  (float16, PyTorch TLS default); DataLoader workers not tested.
* Found: stale AOTAutograd cache can return float16 compiled graphs after the
  change (cache key ignores autocast dtype for `_enter_autocast(..., None, ...)`).
  Follow-up candidate.
* PR + issue draft: coord/tasks/autocast-bf16-default-pr.md
