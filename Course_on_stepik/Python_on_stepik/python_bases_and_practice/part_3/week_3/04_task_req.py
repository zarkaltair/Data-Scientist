import re
import urllib.request
from urllib.parse import urlparse

URL = input()


def get_links():
    try:
        res = urllib.request.urlopen(URL)
        mybytes = res.read()
        mystr = mybytes.decode('utf8')
        res.close()
        links = re.findall(r'<a.+href=[\'"]([^./][^\'"]*)[\'"]', mystr)
    except:
        return []
    else:
        return links


def parse_link(link):
    parsed_uri = urlparse(link)
    res = parsed_uri.netloc
    try:
        return res[:res.index(':')]
    except:
        return res


links = get_links()

unsorted = list(map(parse_link, links))

for link in sorted(list(set(unsorted))):
    print(link)

# import sys
# import re

# sys.stdin = open('04_test.txt', 'r')

# arr_url = [re.findall(r'<a.+href=[\'"]([^./][^\'"]*)[\'"]', line) for line in sys.stdin]

# print(arr_url)