# Issue draft

## Title

Compiled TPU graphs return copies instead of views when no output requires grad

## Body

With `backend="tpu"`, a compiled function that returns a tensor and views of it returns independent copies when no output requires grad.
An in-place write to one output is not seen through the others.
Eager TPU, CPU eager, `aot_eager`, and inductor keep the views.

Repro:

1. Run this script on a TPU host:

   ```python
   import torch
   import torch_tpu  # noqa: F401

   def f(x):
       y = x * 2
       return y, y.view(2, 8), y[1:]

   x = torch.arange(16.0).view(4, 4).to("tpu")
   a, b, c = torch.compile(f, backend="tpu")(x)
   a.add_(100)
   print(b[0, :3].tolist())
   ```

2. Expected: `[100.0, 102.0, 104.0]`, as in eager mode.
3. Actual: `[0.0, 2.0, 4.0]`. `a`, `b`, and `c` do not share storage.

With `x.requires_grad_()` the output is correct, because AOTAutograd regenerates views for outputs that require grad.
For other outputs, AOTAutograd expects the backend to keep the views.

Cause: `_process_fx_outputs` in `src/torch_tpu/_internal/export/export.py` dedups graph outputs only when they are the same view of a buffer.
Every other view becomes a separate graph output with its own buffer.

Proposed fix: return outputs that view the same storage as views of one graph output that covers the full storage.
If no output covers the full storage, add a flat view of the storage as an extra graph output.
Related but different: #6203 (write-back of in-place writes to inputs), #6250 (passthrough elision).

Environment: torch 2.14.1+cpu, torch_tpu 0.3.0 and upstream main (3c6acc43) Python, libtpu 0.0.50, TPU v5e.

# PR draft

## Title

Keep aliasing between outputs of compiled TPU graphs

## Body

## Description

Compiled TPU graphs returned independent copies for outputs that are views of each other when no output requires grad, so an in-place write to one output was not seen through the others.
Eager mode, `aot_eager`, and inductor keep these views.

Outputs that view the same storage are now returned as views of one graph output that covers the full storage.
If no returned output covers it, a flat view of the storage is added as an extra graph output.
This also saves one device buffer per aliased output.
Graphs without aliased outputs keep the C++ `FxOutputReconstructor` fast path.
Bounded-dynamism graphs and views with a different dtype keep the old behavior.

Fixes #<issue>.

## Test Plan

New test `tests/compile/output_alias_test.py` compiles views, slices, transposes, views without their base, `split`, duplicated outputs, and independent outputs, with and without `requires_grad`.
It checks that output aliasing and the values after an in-place write match CPU eager.
It also checks that the new reconstructor survives pickling, for the compile caches.

Run on TPU v5e with torch 2.14.1+cpu, libtpu 0.0.50, torch_tpu 0.3.0 binaries with this branch's Python on `PYTHONPATH` (no bazel build):

| Test (`python3 -m tests.compile.<name>`) | Before | After |
|---|---|---|
| `output_alias_test` (new) | 5 failed, 1 error of 13 | 13 passed |
| `fx_output_reconstructor_test` | 19 passed | 19 passed |
| `small_graph_test` | 75 passed, 7 skipped | 75 passed, 7 skipped |
| `output_modify_test` | 2 passed | 2 passed |
| `clone_mutated_returned_placeholders_test` | 42 passed | 42 passed |
| `backend_serialization_test` | 11 passed | 11 passed |
| `aot_autograd_cache_test` | passed | passed |
| `cache_test` | passed | passed |
| `split_compile_test` | 17 passed, 1 skipped | 17 passed, 1 skipped |

After the fix, `random_ops_test`, `trace_autograd_ops_test`, `activation_checkpointing_test`, and `dtensor_test` pass.
`compile_api_test` passes except 2 `optimization_barrier` gradient tests, which fail the same way without this change in this environment.

`BASE_SHA=upstream/main nox -s lint` passes.
