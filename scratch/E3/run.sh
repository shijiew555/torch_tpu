# driver: repeat locked chunks until __END__
cd /root/torch_tpu/.claude/tmp/E3
for n in $(seq 1 40); do
  grep -q __END__ out.jsonl 2>/dev/null && break
  flock /root/tpu.lock timeout 280 python worker.py ops.txt done.txt out.jsonl 200 >> worker.log 2>&1
  echo "chunk $n rc=$? $(date -u +%T) done=$(wc -l < done.txt)" >> driver.log
  sleep 5
done
