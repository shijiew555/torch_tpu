#!/usr/bin/env bash
# Saves the torch_tpu Bazel caches and built wheel to Google Drive.
#
# Usage: save_build_cache.sh [WORKTREE]
#   WORKTREE: the checkout that ran the wheel build. It is used only to
#   find the built wheel. Default: /root/torch_tpu/.claude/worktrees/base
#
# Saves:
#   * /root/.cache/bazel-disk  (compiled actions - the important part)
#   * /root/.cache/bazel-repo  (downloaded external dependencies)
#   * bazel-bin/ci/wheel/*.whl (the built wheel, if it exists)
#
# Run it when no bazel build is running, so the cache is consistent.
set -euo pipefail

readonly DEST="/content/drive/MyDrive/torch_tpu_build"
readonly CACHE_ROOT="/root/.cache"
readonly WORKTREE="${1:-/root/torch_tpu/.claude/worktrees/base}"

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "ERROR: Google Drive is not mounted at /content/drive." >&2
  exit 1
fi
if pgrep -f 'bazel(isk)?[^ ]* .*(build|test)' >/dev/null; then
  echo "WARNING: a bazel build may be running. Stop it first for a" \
       "consistent cache, or press Ctrl-C within 10 s." >&2
  sleep 10
fi

dirs=()
for d in bazel-disk bazel-repo; do
  if [[ -d "${CACHE_ROOT}/${d}" ]]; then
    dirs+=("${d}")
  else
    echo "Skip: ${CACHE_ROOT}/${d} does not exist."
  fi
done
if [[ ${#dirs[@]} -eq 0 ]]; then
  echo "ERROR: nothing to save." >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "Size to save:"
du -sh "${dirs[@]/#/${CACHE_ROOT}/}"

# Write to a temporary file first, so an interrupted save does not
# replace a good archive with a broken one.
readonly ARCHIVE="${DEST}/bazel-cache.tar.gz"
readonly TMP="${ARCHIVE}.partial"
echo "Writing ${ARCHIVE} ..."
start=$(date +%s)
tar -C "${CACHE_ROOT}" -cf - "${dirs[@]}" | pigz -1 > "${TMP}"
mv -f "${TMP}" "${ARCHIVE}"
echo "Archive done in $(( $(date +%s) - start )) s:" \
     "$(du -h "${ARCHIVE}" | cut -f1)"

shopt -s nullglob
wheels=("${WORKTREE}"/bazel-bin/ci/wheel/*.whl)
if [[ ${#wheels[@]} -gt 0 ]]; then
  mkdir -p "${DEST}/wheel"
  cp -f "${wheels[@]}" "${DEST}/wheel/"
  echo "Copied wheel(s): ${wheels[*]##*/}"
else
  echo "No wheel found in ${WORKTREE}/bazel-bin/ci/wheel/ (skipped)."
fi

{
  echo "saved_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_commit: $(git -C "${WORKTREE}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "bazel_version: $(cat "${WORKTREE}/.bazelversion" 2>/dev/null || echo unknown)"
  echo "python: $(python3 --version 2>&1)"
  echo "torch: $(pip show torch 2>/dev/null | awk '/^Version/{print $2}')"
} > "${DEST}/MANIFEST.txt"
cat "${DEST}/MANIFEST.txt"
echo "Done. Saved to ${DEST}."
