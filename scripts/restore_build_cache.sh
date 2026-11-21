#!/usr/bin/env bash
# Restores the torch_tpu Bazel caches from Google Drive to a new VM.
#
# Usage: restore_build_cache.sh
#
# Restores /root/.cache/bazel/bazel-disk-cache (the disk cache path that
# the repo .bazelrc sets) and /root/.cache/bazel-repo, and adds the
# repository cache to /root/.bazelrc. Then the build command in
# BUILD_NOTES.md reuses the cache. Cache hits need the same commit, Bazel
# version, and build flags as in MANIFEST.txt.
set -euo pipefail

readonly SRC="/content/drive/MyDrive/torch_tpu_build"
readonly CACHE_ROOT="/root/.cache"
readonly ARCHIVE="${SRC}/bazel-cache.tar.gz"
readonly BAZELRC="/root/.bazelrc"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "ERROR: ${ARCHIVE} not found. Is Google Drive mounted?" >&2
  exit 1
fi
[[ -f "${SRC}/MANIFEST.txt" ]] && { echo "Saved build:"; cat "${SRC}/MANIFEST.txt"; }

mkdir -p "${CACHE_ROOT}"
echo "Extracting $(du -h "${ARCHIVE}" | cut -f1) to ${CACHE_ROOT} ..."
start=$(date +%s)
pigz -dc "${ARCHIVE}" | tar -C "${CACHE_ROOT}" -xf -
echo "Extract done in $(( $(date +%s) - start )) s."

touch "${BAZELRC}"
for line in "common --repository_cache=${CACHE_ROOT}/bazel-repo"; do
  grep -qxF "${line}" "${BAZELRC}" || echo "${line}" >> "${BAZELRC}"
done
echo "${BAZELRC}:"
cat "${BAZELRC}"

if ! command -v bazel >/dev/null; then
  echo "Installing bazelisk as /usr/local/bin/bazel ..."
  wget -q -O /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
  chmod +x /usr/local/bin/bazel
fi

echo "Done. Build with the command in BUILD_NOTES.md."
