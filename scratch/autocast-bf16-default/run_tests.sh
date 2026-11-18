#!/bin/bash
# Usage: run_tests.sh <repo> <overlay> <tag>
E=/root/torch_tpu/.claude/tmp/autocast-bf16-default
cd $E/$1
for t in ${TESTS:-tests.amp_test tests.device.device_module_tpu_test}; do
  rm -rf $E/cache_$3
  TORCHINDUCTOR_CACHE_DIR=$E/cache_$3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$E/$2 \
    flock /root/tpu.lock timeout 1200 python3 -m $t > $E/log_$3_$t.txt 2>&1
  echo "$3 $t exit=$? $(grep -E '^(Ran|OK|FAILED)' $E/log_$3_$t.txt | tr '\n' ' ')"
done
