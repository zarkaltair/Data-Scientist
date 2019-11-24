import re
import sys

sys.stdin = open('09_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    dupl = re.sub(r'(\w)\1+', r'\1', line)
    print(dupl)
