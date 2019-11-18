import sys
from datetime import date
from datetime import timedelta

sys.stdin = open('test_time.txt', 'r')

d = input().split(' ')
x = input()

new_date = date(int(d[0]), int(d[1]), int(d[2])) + timedelta(int(x))
str_date = str(new_date).split('-')
arr = [i if i[0] != '0' else i[1:] for i in str_date]
cor_new_date = ' '.join(arr)
print(new_date)
print(type(new_date))
print(type(cor_new_date))
print(cor_new_date)
print(str_date)