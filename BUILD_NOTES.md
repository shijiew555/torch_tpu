# Building torch_tpu from source on the Colab TPU VM

Status: in progress, not yet verified to finish.
At 2026-11-18 04:30 UTC the build was at 17,761 of 20,617 actions with no errors.
This file will be updated when the build ends.

## Environment

* Colab TPU VM, 24 CPUs, 47 GB RAM, 226 GB disk.
* Python 3.13, torch 2.14.1+cpu, torch_tpu 0.3.0 wheel installed in system Python.
* Source: upstream `google-pytorch/torch_tpu` main at `afd6943a`.
* Bazel 9.3.1 through bazelisk.

## One-time setup

1. Install bazelisk as `bazel`:

   ```bash
   wget -q -O /usr/local/bin/bazel \
     https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
   chmod +x /usr/local/bin/bazel
   ```

2. Install nox (for `nox -s lint` and `nox -s format`):

   ```bash
   pip install nox
   ```

3. Share build caches between all worktrees.
   Add these lines to `/root/.bazelrc`:

   ```
   common --disk_cache=/root/.cache/bazel-disk
   common --repository_cache=/root/.cache/bazel-repo
   ```

   The caches are not persistent on Colab.
   A new VM starts the build from zero.

## Build command

Run this from the repo root (or a worktree of upstream main):

```bash
bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel \
  --config=no_rbe \
  --spawn_strategy=sandboxed,local --strategy=sandboxed,local
```

* `--config=no_rbe` is needed because there are no remote build credentials.
  It must come after `--config=wheel_common`.
* The two strategy flags are the workaround from upstream issue #6508.
  They must come after `--config=no_rbe`.
* Do not add `--config=testone`.
  It turns unsandboxed execution back on and undoes the workaround.
* Run the build in the background with a log file.

The wheel is written to `bazel-bin/ci/wheel/`.

## Known failure without the workaround

Without the strategy flags, the build fails after about 15 minutes, at about 12,700 actions.
The error is in grpc:

```
external/grpc+/src/core/channelz/v2tov1/property_list.cc:22:10: error:
module @@grpc+//src/core:channelz_v2tov1_property_list does not depend on
a module exporting 'google/protobuf/any.upb.h'
```

This is the first error in upstream issue #6508.
The cause is that `common:no_rbe` pulls in `common:local`, which sets `--spawn_strategy=standalone`.
Actions then run without a sandbox and see headers that are not declared.

## Timing

* Start: 03:01 UTC. Total: 20,617 actions.
* 03:19 UTC: 16,320 actions. About 24 actions per minute while it compiled dependencies.
* 04:30 UTC: 17,761 actions. About 11 actions per minute while it compiled `csrc/`.
  Some single `csrc/` files compile for more than 45 minutes.
* Other TPU jobs on the same machine slow the build down.

## Optional remote cache (not tested)

`--config=helion_public_caching_readonly` can be read without credentials.
Put it after `--config=no_rbe`, because `no_rbe` clears `--remote_cache`.
Helion CI writes this cache with Python 3.12 and `local_torch`, so many actions may miss.

## Running one test against the built wheel (not yet run)

```bash
bazel test --config=wheel_common --config=no_rbe \
  --spawn_strategy=sandboxed,local --strategy=sandboxed,local \
  --//:test_wheel=True //tests:env_utils_test
```

For TPU tests, build first without the TPU lock.
Then run `bazel test` under `flock /root/tpu.lock` with `--local_test_jobs=1`.

## Testing Python-only changes without a build

The installed 0.3.0 binaries can run the upstream main Python code through a `PYTHONPATH` overlay.
See `discovery/E6.md` for the exact steps.
