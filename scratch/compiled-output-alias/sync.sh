#!/bin/bash
# Copy worktree Python sources and tests into the overlay and repo copy.
set -e
E=/root/torch_tpu/.claude/tmp/compiled-output-alias
W=/root/torch_tpu/.claude/worktrees/compiled-output-alias
cd $W/src/torch_tpu && find . -name '*.py' -exec cp --parents {} $E/ov/torch_tpu/ \;
rm -rf $E/repo && mkdir -p $E/repo && cp -r $W/tests $E/repo/ && cp -r $W/perflab $E/repo/ 2>/dev/null || true
