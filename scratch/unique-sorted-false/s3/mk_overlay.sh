# Builds $E/ov: installed torch_tpu (.so as symlinks) plus worktree Python source.
set -e
rm -rf /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov; mkdir -p /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov/torch_tpu
cd /root/torch_tpu/.claude/worktrees/unique-sorted-false/.venv/lib/python3.14/site-packages/torch_tpu
find . -type d -exec mkdir -p /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov/torch_tpu/{} \;
find . -type f ! -name '*.so*' -exec cp -p {} /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov/torch_tpu/{} \;
find . -type f -name '*.so*' -exec ln -s /root/torch_tpu/.claude/worktrees/unique-sorted-false/.venv/lib/python3.14/site-packages/torch_tpu/{} /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov/torch_tpu/{} \;
cd /root/torch_tpu/.claude/worktrees/unique-sorted-false/src/torch_tpu && find . -name '*.py' -exec cp --parents {} /root/torch_tpu/.claude/tmp/unique-sorted-false-s3/ov/torch_tpu/ \;
