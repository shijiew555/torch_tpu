cd /root/torch_tpu/.claude/tmp/E3
for f in repro_lcse_grad.py; do
  echo "=== $f"; timeout 60 python $f 2>&1 | grep -v -E "^I0000|^W0000|^WARNING|tcmalloc|PrivateUse1|Registered Python|Device type|Initializing TPU|^num_|^peak_|Warning|^  return|^  warn"
done
