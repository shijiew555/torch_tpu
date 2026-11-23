# compiled-output-alias (worker 3)

* 07:20Z Worktree at upstream/main 3c6acc43. Repro on overlay confirms bug (backend=tpu no-grad row: shared=False). Log: .claude/tmp/compiled-output-alias/repro_before.log
* Plan: Python-only. In `_process_fx_outputs`, group outputs by storage; emit one flat storage-wide output per group; rebuild members with `as_strided` in a Python reconstructor that wraps the C++ one. Graphs without aliased outputs keep the C++ fast path.
* New test: tests/compile/output_alias_test.py
* 07:20Z New test fails before fix: 5 of 12 cases (views+slice, transpose, views-only, split no-grad, split grad). Log test_before.log.
* 07:22Z First fix (flat storage base for all groups) broke training graphs: output y became a view made inside the autograd Function, so in-place writes then reads raise. Fixed: when an output covers the full storage, it is the base (plain tensor); a hidden flat base is used only for views-only groups. Test now reads outputs with autograd checks (mutant check: forced flat base makes 3 grad cases ERROR).
* 07:23Z Commit e1d54824 pushed. Existing compile test suite before/after running.
* 07:40Z Existing compile tests (9 modules) before/after: all pass, no change. New test: before 5/12 fail, after 12/12 pass. Lint passes (commit 9858e789). Review pass done, no further changes. Final validation (final test file before/after + 5 more compile modules after) running.
* 08:09Z Done. Backup `compiled-output-alias` 9244adbe. PR branch `compiled-output-alias-pr` 0fc40a74 (squash 424a80b4 + pickle-test fix 0fc40a74; no force-push). Lint passes on both.
* Final: new test 13/13 after (before: 5 fail + 1 error). Existing 9 modules unchanged. compile_api_test: 2 optimization_barrier tests fail before and after (overlay env, not this change).
* Draft: tasks/compiled-output-alias-pr.md (issue + PR).

## Session 3 refresh (worker, started 23:29Z)

* Worktree .claude/worktrees/compiled-output-alias (branch compiled-output-alias 9244adbe), venv .venv: Python 3.14.9, afd6943a wheel, torch 2.16.0.dev20261112+cpu, libtpu 0.0.53.1. Overlay script and logs: .claude/tmp/compiled-output-alias3/.
* E2E repro (overlay of 3c6acc43 vs fix): base `b[0,:3]=[0,2,4]`, shared=False; fix `[100,102,104]`, shared=True.
* PR #6203 touches export.py in another region. `git merge-tree` of pr2 with #6203: export.py merges cleanly. Its 2 conflicts (clone_mutated_returned_placeholders.py, compile_api_test.py) also occur between #6203 and upstream/main itself. It does not fix this bug. No upstream commit since 3c6acc43 changes export.py.
* New PR branch compiled-output-alias-pr2 4b53310f (1 commit on upstream/main 82d0a494), pushed. Lint and format pass on it.
* 23:50Z Review pass found a bug: groups keyed by `data_ptr()`, which is 0 for tensors with no elements, so `y[:0]` grouped with an unrelated `new_zeros(0)`. Fix: key by `untyped_storage().data_ptr()`, skip 0-byte storages. New test case `_empty_slice` failed before, passes after. Commit f7ee0c2f on backup and pr2 (pr2 = 4b53310f + f7ee0c2f, normal push). Lint and format pass.
* Overlay caveat: upstream 1694da75 makes compiler.py call C++ `tpu_torch_compile.get_non_default_device_layout`, missing in the afd6943a wheel. Plain main/pr2 overlays fail with AttributeError. Test overlays `ov_mainc`/`ov_pr2c` revert 1694da75 in compiler.py only.
* 00:05Z Results on main 82d0a494 vs pr2 f7ee0c2f (summary3.txt, summary4.txt): output_alias_test 6 failed + 1 error of 15 -> 15 OK. 12 other compile modules identical on both. backend_serialization_test float4 cache test fails on both; compile_api_test aborts on both at test_force_strides_strided_middle_dimension (61 OK before it). Environment, not this change.
* Default /tmp inductor/AOTAutograd cache ignores overlay Python changes; always use a fresh TORCHINDUCTOR_CACHE_DIR.
* State: done. PR draft updated: pr_desc/compiled-output-alias.md.
