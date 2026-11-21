# Building torch_tpu from source on the Colab TPU VM

Status: verified.
The build of upstream main `afd6943a` finished successfully at 2026-11-18 06:51 UTC.
Elapsed time: 14,584 s (about 4 hours) without a remote cache, on a shared machine.
Output: `bazel-bin/ci/wheel/torch_tpu-0.3.0-cp314-cp314-manylinux_2_31_x86_64.whl` (107 MB).
The wheel is for Python 3.14 (the hermetic Python of the build), not for the system Python 3.13.
Repository cache size: 12 GB.

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

3. Share the repository cache between all worktrees.
   Add this line to `/root/.bazelrc`:

   ```
   common --repository_cache=/root/.cache/bazel-repo
   ```

   The disk cache needs no setup.
   The repo `.bazelrc` already sets `build --disk_cache=~/.cache/bazel/bazel-disk-cache`, and all worktrees share it.
   A `--disk_cache` line under `common` in `/root/.bazelrc` has no effect, because the repo's `build` line overrides it.
   The caches are not persistent on Colab.
   A new VM starts the build from zero unless you restore them (see below).

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

## Save and restore the build cache (Google Drive)

Two scripts in `scripts/` keep the build results across Colab VMs.
A full build takes hours, but a restored cache turns most of it into cache hits.
Google Drive must be mounted at `/content/drive`.

### What is saved

All files go to `/content/drive/MyDrive/torch_tpu_build/`:

* `bazel-cache.tar.gz`: the Bazel disk cache (`/root/.cache/bazel/bazel-disk-cache`, 5.8 GB on 2026-11-18) and repository cache (`/root/.cache/bazel-repo`, 12 GB).
  The disk cache holds the compiled actions and is the important part.
* `wheel/*.whl`: the built wheel, if the build made one.
* `MANIFEST.txt`: the save time, source commit, and Bazel, Python, and torch versions.

The Bazel output bases (`/root/.cache/bazel/_bazel_root/`) are not saved.
It is very large, and Bazel fills it again from the disk cache.

### Save (before the VM stops)

1. Wait until no bazel build or test is running.
   If one is running, the script warns you and waits 10 seconds, so that you can stop it with Ctrl-C.
2. Run:

   ```bash
   bash /root/torch_tpu/.claude/coord/scripts/save_build_cache.sh
   ```

   To take the wheel from a different checkout, give its path:

   ```bash
   bash /root/torch_tpu/.claude/coord/scripts/save_build_cache.sh \
     /root/torch_tpu/.claude/worktrees/<name>
   ```

3. Check the output.
   It shows the size, the time it took, the wheel name, and the manifest.

The script writes `bazel-cache.tar.gz.partial` first and renames it at the end.
So an interrupted save does not replace an older good archive.
If a save is interrupted, delete the `.partial` file.

### Restore (on a new VM)

1. Clone the fork and get this branch:

   ```bash
   git clone https://github.com/shijiew555/torch_tpu.git /root/torch_tpu
   git -C /root/torch_tpu fetch origin agent-coord
   ```

2. Run the restore script straight from the branch:

   ```bash
   git -C /root/torch_tpu show origin/agent-coord:scripts/restore_build_cache.sh | bash
   ```

   To also install the saved wheel into system Python, add the flag:

   ```bash
   git -C /root/torch_tpu show origin/agent-coord:scripts/restore_build_cache.sh \
     | bash -s -- --install-wheel
   ```

3. Make a worktree at the commit in `MANIFEST.txt` (`source_commit`):

   ```bash
   git -C /root/torch_tpu remote add upstream \
     https://github.com/google-pytorch/torch_tpu.git
   git -C /root/torch_tpu fetch upstream main
   git -C /root/torch_tpu worktree add --detach \
     .claude/worktrees/build-base <source_commit>
   ```

   Do not build the fork's `main`.
   It can be an older commit with a different XLA revision (`bazel/xla_revision.bzl`).
   A different XLA revision makes all of XLA compile again, which takes hours.

4. In the worktree, run the build command from "Build command" above.
   Bazel uses the restored cache, so only the changed files compile.
   If the first try fails with `Timed out` on the torch download, see "Known failure on the first build in a new output base".

The restore script does these steps:

* Shows `MANIFEST.txt`.
* Unpacks the caches into `/root/.cache/`.
* Adds the `--repository_cache` line to `/root/.bazelrc`, once only.
* Installs bazelisk as `/usr/local/bin/bazel` if it is missing.
* With `--install-wheel`, runs `pip install --force-reinstall --no-deps` on the saved wheel.

### Evidence that the cache works

On 2026-11-18, new worktrees on the same VM reused the disk cache of the base build.
Worker builds in new worktrees had about 13,600 disk cache hits each.
Each worktree has its own output base, so a new VM with a restored cache is the same case.

On 2026-11-21, a restore on a new VM worked.
The build of `afd6943a` had 13,713 disk cache hits, 6,904 internal actions, and 0 local compiles.
The successful try took 1,205 s.
The total, with the failed first try, was about 33 minutes.
Most of the time is setup in the new output base: Bazel unpacks the external repositories (LLVM is the slowest) and downloads the torch nightly wheel.
A second build in the same worktree skips this setup.

### When the cache does not help

* Cache hits need the same Bazel, Python, and torch versions as in `MANIFEST.txt`, and the same build flags.
* Files that changed since the saved commit compile again.
  If upstream main moved a lot, more of the build runs again.
* A different XLA revision in `bazel/xla_revision.bzl` makes all of XLA and everything that depends on it compile again.
  Compare it with the saved commit before you build another commit.
* The disk cache stores only action outputs.
  It does not store analysis results or unpacked repositories, so a new output base always spends time on setup.
* The saved wheel matches only the commit in `MANIFEST.txt`.
  Do not use it to test a change on a newer commit.

### Test status

* Save: tested on 2026-11-18.
  It took 1,479 s and wrote a 9.5 GB archive and the 107 MB base wheel.
  `pigz -t` on the archive and a zip test on the wheel both passed.
* Restore on a new VM: tested on 2026-11-21.
  The build of the saved commit had 0 local compiles (see "Evidence that the cache works").
