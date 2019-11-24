import re
import sys

sys.stdin = open('03_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    zz = re.findall(r'z[a-z]{3}z', line)
    if zz != []:
        print(line)
