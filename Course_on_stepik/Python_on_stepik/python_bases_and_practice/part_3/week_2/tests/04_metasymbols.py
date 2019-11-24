import re
import sys

sys.stdin = open('04_test.txt', 'r')

for line in sys.stdin:
    line = line.rstrip()
    meta_symb = re.findall(r'\\', line)
    if meta_symb != []:
        print(line)
