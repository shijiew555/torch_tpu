#!/bin/bash
# Usage: make_overlay.sh <git-ref> <out-dir>
# Builds a torch_tpu overlay: installed afd6943a wheel binaries + Python
# sources from <git-ref>, and a repo-tree copy of tests/ for running tests.
set -euo pipefail
REF=$1; OUT=$(realpath -m "$2")
W=/root/torch_tpu/.claude/worktrees/ckpt-tpu-rng-state
I=$W/.venv/lib/python3.14/site-packages/torch_tpu
rm -rf "$OUT"; mkdir -p "$OUT/ov/torch_tpu" "$OUT/src" "$OUT/repo"
cd "$I"
find . -type d -exec mkdir -p "$OUT/ov/torch_tpu/{}" \;
find . -type f ! -name '*.so*' -exec cp -p {} "$OUT/ov/torch_tpu/{}" \;
find . -type f -name '*.so*' -exec ln -s "$I/{}" "$OUT/ov/torch_tpu/{}" \;
git -C $W archive "$REF" src/torch_tpu | tar -x -C "$OUT/src"
cd "$OUT/src/src/torch_tpu"
find . -name '*.py' -exec install -D -m 644 {} "$OUT/ov/torch_tpu/{}" \;
git -C $W archive "$REF" tests | tar -x -C "$OUT/repo"
echo "overlay ready: $OUT"
