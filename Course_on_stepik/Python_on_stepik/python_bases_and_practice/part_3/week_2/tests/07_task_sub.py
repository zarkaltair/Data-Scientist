import re
import sys

sys.stdin = open('07_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    substr = re.sub(r'\ba+\b', 'argh', line, 1, re.IGNORECASE)
    print(substr)
