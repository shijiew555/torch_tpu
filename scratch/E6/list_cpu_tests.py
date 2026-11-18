"""Lists torch_tpu_py_test targets in a BUILD file that have no requires-tpu tag."""
import re, sys
text = open(sys.argv[1]).read()
for m in re.finditer(r'torch_tpu_py_test\(\s*name = "([^"]+)",(.*?)\n\)', text, re.S):
  name, body = m.group(1), m.group(2)
  if 'requires-tpu' in body or 'REQUIRES_V5' in body or 'run_on_accelerators' in body or 'is_wheel_test' in body:
    continue
  print(name, 'manual' if 'manual' in body or 'notest_oss' in body else '')
