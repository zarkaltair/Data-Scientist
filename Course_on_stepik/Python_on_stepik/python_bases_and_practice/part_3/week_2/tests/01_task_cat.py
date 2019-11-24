import re
import sys

sys.stdin = open('01_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    cat = re.findall('cat', line)
    if len(cat) > 1:
        print(line)