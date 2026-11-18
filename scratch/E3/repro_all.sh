cd /root/torch_tpu/.claude/tmp/E3
for f in repro_unique.py repro_trilinear.py repro_kthvalue.py repro_pad.py; do
  echo "=== $f"; timeout 60 python $f 2>&1 | grep -v -E "^I0000|^W0000|^WARNING|tcmalloc|PrivateUse1|Registered Python|Device type|Initializing TPU"
done
