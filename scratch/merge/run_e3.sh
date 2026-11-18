cd /root/torch_tpu/.claude/tmp/E3
for f in repro_unfold_grad.py repro_fold.py repro_unique.py repro_amp_grad.py; do
  echo "== $f"
  flock /root/tpu.lock timeout 150 python $f > /root/torch_tpu/.claude/tmp/merge/$f.log 2>&1
  echo "exit=$?"
  grep -vE "Warning|^num_|^peak|^$|warn" /root/torch_tpu/.claude/tmp/merge/$f.log | tail -4
done
