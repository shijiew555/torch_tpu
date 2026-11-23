#!/bin/bash
# Usage: mkov.sh <git-ref> <name>. Builds overlay $E/ov_<name> (installed
# afd6943a package with all .py from <ref>:src/torch_tpu) and repo copy
# $E/repo_<name> (tests from <ref>).
set -e
E=/root/torch_tpu/.claude/tmp/compiled-output-alias3
I=/root/torch_tpu/.claude/worktrees/compiled-output-alias/.venv/lib/python3.14/site-packages/torch_tpu
O=$E/ov_$2/torch_tpu
rm -rf $E/ov_$2 $E/repo_$2 && mkdir -p $O $E/repo_$2 $E/src_$2
cd $I
find . -type d ! -name __pycache__ -exec mkdir -p $O/{} \;
find . -type f ! -name '*.so*' ! -path '*__pycache__*' -exec cp -p {} $O/{} \;
find . -type f -name '*.so*' -exec ln -s $I/{} $O/{} \;
rm -rf $E/src_$2 && mkdir -p $E/src_$2
git -C /root/torch_tpu archive $1 src/torch_tpu | tar -x -C $E/src_$2
(cd $E/src_$2/src/torch_tpu && find . -name '*.py' -exec cp --parents {} $O/ \;)
git -C /root/torch_tpu archive $1 tests perflab | tar -x -C $E/repo_$2
echo "overlay $2 from $(git -C /root/torch_tpu rev-parse --short $1)"
