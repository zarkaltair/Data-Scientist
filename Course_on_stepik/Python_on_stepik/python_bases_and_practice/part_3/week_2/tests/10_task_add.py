import re
import sys

sys.stdin = open('10_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    match = re.match(r'^(0|(1(01*0)*10*)+)$', line)
    print(match)
