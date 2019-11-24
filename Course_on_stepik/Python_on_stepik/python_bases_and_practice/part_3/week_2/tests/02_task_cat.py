import re
import sys

sys.stdin = open('02_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    cat = re.findall(r'\bcat\b', line)
    if len(cat) >= 1:
        print(line)