# Usage: mk.sh <name> <git-ref>. Builds overlay ov_<name> and repo_<name>.
set -e
E=/root/torch_tpu/.claude/tmp/autocast-bf16-default-s3
I=/root/torch_tpu/.claude/worktrees/autocast-bf16-default/.venv/lib/python3.14/site-packages/torch_tpu
N=$1; R=$2
rm -rf $E/ov_$N $E/repo_$N $E/src_$N; mkdir -p $E/ov_$N/torch_tpu $E/repo_$N $E/src_$N
cd $I
find . -type d -exec mkdir -p $E/ov_$N/torch_tpu/{} \;
find . -type f ! -name '*.so*' -exec cp -p {} $E/ov_$N/torch_tpu/{} \;
find . -type f -name '*.so*' -exec ln -s $I/{} $E/ov_$N/torch_tpu/{} \;
git -C /root/torch_tpu archive $R src/torch_tpu | tar -x -C $E/src_$N
cd $E/src_$N/src/torch_tpu && find . -name '*.py' -exec cp --parents {} $E/ov_$N/torch_tpu/ \;
git -C /root/torch_tpu archive $R tests perflab | tar -x -C $E/repo_$N
echo built $N $(git -C /root/torch_tpu rev-parse --short $R)
