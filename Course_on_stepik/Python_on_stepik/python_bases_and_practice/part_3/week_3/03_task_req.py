import sys
import re
import requests

sys.stdin = open('03_test.txt', 'r')

def find_all(res):
    return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ])+', res.text)

url_1, url_2 = input(), input()

res_1 = requests.get(url_1)
find_url_1 = find_all(res_1)

arr_url = []
for i in find_url_1:
    res = requests.get(i)
    arr_url += find_all(res)

if url_2 in arr_url:
    print('Yes')
else:
    print('No')

# pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
