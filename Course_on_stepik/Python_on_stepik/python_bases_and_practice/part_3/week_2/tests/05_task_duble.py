import re
import sys

sys.stdin = open('05_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    dupl = re.findall(r'\b(\w+)\1\b', line)
    if dupl != []:
        print(line)
