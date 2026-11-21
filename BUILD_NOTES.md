# Building torch_tpu from source on the Colab TPU VM

Status: verified.
The build of upstream main `afd6943a` finished successfully at 2026-11-18 06:51 UTC.
Elapsed time: 14,584 s (about 4 hours) without a remote cache, on a shared machine.
Output: `bazel-bin/ci/wheel/torch_tpu-0.3.0-cp314-cp314-manylinux_2_31_x86_64.whl` (107 MB).
The wheel is for Python 3.14 (the hermetic Python of the build), not for the system Python 3.13.
Repository cache size: 12 GB.

## Environment

* Colab TPU VM, 24 CPUs, 47 GB RAM, 226 GB disk.
* System Python 3.13, torch 2.14.1+cpu.
  A new VM can have an older torch_tpu.
  To match these notes, run `pip install torch_tpu==0.3.0`.
* The build uses its own hermetic Python 3.14 and torch nightly, not the system Python.
* Source: upstream `google-pytorch/torch_tpu` main at `afd6943a`.
* Bazel 9.3.1 through bazelisk.

## One-time setup

On a new VM with a saved cache, the restore script does steps 1 and 3.
See "Restore on a new VM".

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

3. Share the repository cache between all worktrees.
   Add this line to `/root/.bazelrc`:

   ```
   common --repository_cache=/root/.cache/bazel-repo
   ```

   The disk cache needs no setup.
   The repo `.bazelrc` already sets `build --disk_cache=~/.cache/bazel/bazel-disk-cache`, and all worktrees share it.
   Do not add a `--disk_cache` line under `common` in `/root/.bazelrc`.
   It has no effect, because the repo's `build` line overrides it.

## Build command

Run this from the repo root (or a worktree of upstream main).
Bazel needs `MODULE.bazel` in the current directory.
If you run it from `/root`, it fails.

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

## Known failure on the first build in a new output base

On a new VM or in a new worktree, the first analysis can fail after about 12 minutes with this error:

```
rules_python:pip_archive(@@rules_python++pip+torch_tpu_pypi_2_16_0_314_torch) FAIL: ...
torch==2.16.0.dev20261112+cpu): end: failure:
...
===== stderr start =====
Timed out
```

* The cause is the 600 s limit on the download of the torch nightly wheel (222 MB) for the hermetic Python 3.14.
  The download runs at the same time as the unpack of the LLVM toolchain, and it can be too slow.
* The wheel is still on the PyTorch nightly index, so this is not a missing file.
* To fix it, run the same build command again.
  Repositories that finished stay in the output base, so the second analysis is faster.
  On 2026-11-21 the second try passed.

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

## Running one test against the built wheel

On 2026-11-20, `//tests:async_transfer_test` passed on TPU this way (see `tasks/env-build.md`).

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

## Save and restore the build cache (Google Drive)

A cold build takes about 4 hours.
With a restored cache, a build of the saved commit on a new VM takes about 30 minutes.
Two scripts on this branch save the cache to Google Drive and restore it.
Google Drive must be mounted at `/content/drive`.

### What is saved

The save script writes these files to `/content/drive/MyDrive/torch_tpu_build/`:

* `bazel-cache.tar.gz`: one archive of the two Bazel caches (9.5 GB compressed on 2026-11-18).
  * Disk cache, `/root/.cache/bazel/bazel-disk-cache` (5.8 GB).
    It holds the outputs of build actions, such as object files and linked `.so` files.
    It saves the compile time, so it is the important part.
    The repo `.bazelrc` sets this path with `build --disk_cache`, so it needs no setup.
  * Repository cache, `/root/.cache/bazel-repo` (12 GB).
    It holds the downloaded archives of external repositories, such as XLA and LLVM.
    It saves the download time.
    `/root/.bazelrc` sets this path with `common --repository_cache`.
* `wheel/*.whl`: the wheel of the saved commit, if the build made one.
* `MANIFEST.txt`: the save time, the source commit, the Bazel version, and the XLA revision.

These things are not saved, so a new VM spends about 15 to 20 minutes on setup:

* The Bazel output base (`/root/.cache/bazel/_bazel_root/<hash>/`).
  Bazel unpacks the external repositories into it again from the repository cache.
  LLVM is the slowest.
* The torch nightly wheel of the hermetic Python 3.14 (222 MB).
  rules_python downloads it with pip, not through the repository cache.

### When to save

* Save after a build at a new base commit, for example after a rebase that changes `MODULE.bazel` or `bazel/xla_revision.bzl`.
* You do not need to save after small changes to `csrc/` or Python files.
  Bazel compiles only those files again, on top of the saved cache.
* A save takes about 25 minutes, so start it well before the VM stops.

### Save

1. Make sure that no `bazel build` or `bazel test` runs.
   If one runs, the script warns you and waits 10 seconds, so that you can stop it with Ctrl-C.
2. Run the save script from this branch.
   Give the worktree that ran the build.
   The script takes the wheel and the source commit from it.

   ```bash
   git -C /root/torch_tpu fetch origin agent-coord
   git -C /root/torch_tpu show origin/agent-coord:scripts/save_build_cache.sh \
     | bash -s -- /root/torch_tpu/.claude/worktrees/build-base
   ```

3. Check the output.
   It shows the size, the time, the wheel name, and the manifest.

The script writes `bazel-cache.tar.gz.partial` first and renames it at the end.
So an interrupted save does not replace an older good archive.
If a save is interrupted, delete the `.partial` file.

### Restore on a new VM

1. Mount Google Drive at `/content/drive`.
2. Clone the fork, and get this branch and upstream main:

   ```bash
   git clone https://github.com/shijiew555/torch_tpu.git /root/torch_tpu
   cd /root/torch_tpu
   git remote add upstream https://github.com/google-pytorch/torch_tpu.git
   git fetch origin agent-coord
   git fetch upstream main
   ```

3. Run the restore script from this branch:

   ```bash
   git -C /root/torch_tpu show origin/agent-coord:scripts/restore_build_cache.sh \
     | bash
   ```

   The script does these steps:

   * Shows `MANIFEST.txt`.
   * Unpacks the two caches into `/root/.cache/`.
   * Adds the `--repository_cache` line to `/root/.bazelrc`, once only.
   * Installs bazelisk as `/usr/local/bin/bazel` if it is missing.

4. Make a worktree at the `source_commit` in `MANIFEST.txt`:

   ```bash
   cd /root/torch_tpu
   git worktree add --detach .claude/worktrees/build-base <source_commit>
   ```

   Do not build the fork's `main`.
   It can be a different commit with a different XLA revision.
   A different XLA revision makes all of XLA compile again, which takes hours.

5. In the worktree, run the command from "Build command":

   ```bash
   cd /root/torch_tpu/.claude/worktrees/build-base
   bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel \
     --config=no_rbe \
     --spawn_strategy=sandboxed,local --strategy=sandboxed,local
   ```

   If the first try fails with `Timed out` on the torch download, run it again.
   See "Known failure on the first build in a new output base".

6. To develop, make a branch in the same worktree.
   Keep using this worktree, because a new worktree has a new output base and spends the setup time again.

### When the cache does not help

* Cache hits need the same Bazel version and the same build flags.
* Files that changed since the saved commit compile again.
  A change to a header that many files include makes all of those files compile again.
* A different XLA revision in `bazel/xla_revision.bzl` makes all of XLA and everything that depends on it compile again.
  Compare it with `MANIFEST.txt` before you build another commit.
* The saved wheel matches only the commit in `MANIFEST.txt`.
  It is for Python 3.14, so you cannot install it into the system Python 3.13.
  Use it through `bazel test --//:test_wheel=True`, or in a Python 3.14 venv.

### Test results

* Save, 2026-11-18: 1,479 s.
  It wrote a 9.5 GB archive and the 107 MB wheel.
  `pigz -t` on the archive and a zip test on the wheel both passed.
* Restore on a new VM, 2026-11-21: the build of `afd6943a` had 13,713 disk cache hits, 6,904 internal actions, and 0 local compiles.
  The build that passed took 1,205 s.
  With the first try that timed out, the total was about 33 minutes.
