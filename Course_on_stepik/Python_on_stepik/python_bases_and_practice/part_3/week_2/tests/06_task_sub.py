import re
import sys

sys.stdin = open('06_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    substr = re.sub('human', 'computer', line)
    print(substr)
