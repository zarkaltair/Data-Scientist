import sys


def summ(num):
    return eval('+'.join([str(i) for i in str(num)]))

digit_string = sys.argv[1]


print(summ(digit_string))
