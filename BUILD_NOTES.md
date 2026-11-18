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

3. Run the build command from "Build command" above.
   Bazel uses the restored cache, so only the changed files compile.

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
A restore on a new VM was not tested yet.

### When the cache does not help

* Cache hits need the same Bazel, Python, and torch versions as in `MANIFEST.txt`, and the same build flags.
* Files that changed since the saved commit compile again.
  If upstream main moved a lot, more of the build runs again.
* The saved wheel matches only the commit in `MANIFEST.txt`.
  Do not use it to test a change on a newer commit.

### Test status

Only the bash syntax and the running-build check were tested.
A full save and restore was not run yet.
