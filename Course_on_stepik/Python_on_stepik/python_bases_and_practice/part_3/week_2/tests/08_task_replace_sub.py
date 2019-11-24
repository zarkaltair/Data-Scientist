import re
import sys

sys.stdin = open('08_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    match = re.sub(r'\b(\w)(\w)', r'\2\1', line)
    print(match)
