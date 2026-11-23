E=/root/torch_tpu/.claude/tmp/compiled-output-alias3
PY=/root/torch_tpu/.claude/worktrees/compiled-output-alias/.venv/bin/python
cd $E
for v in base fix; do
  echo "== $v"
  rm -rf $E/cache_$v
  TORCHINDUCTOR_CACHE_DIR=$E/cache_$v PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$E/ov_$v timeout 600 $PY repro.py 2>&1 | grep -E '^(torch_tpu from|b\[0|shared|Traceback|\w+Error)'
done
