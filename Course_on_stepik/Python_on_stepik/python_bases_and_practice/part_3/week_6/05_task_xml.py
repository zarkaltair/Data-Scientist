import sys
from lxml import etree
from xml.etree import ElementTree

sys.stdin = open('05_test.xml')

def get_color(el, x=1):
    if el.attrib['color'] == 'blue':
        dct[el.attrib['color']] += x
    elif el.attrib['color'] == 'red':
        dct[el.attrib['color']] += x
    else:
        dct[el.attrib['color']] += x
    x += 1
    for i in el.getchildren():
        get_color(i, x)

root = etree.fromstring(input())
dct = {i: 0 for i in ('blue', 'red', 'green')}
get_color(root)

arr = sorted(dct.items(), key=lambda x: (x[1], x[0]))
for i in arr[::-1]:
    print(i[1], end=' ')


# stack for tree
stack = [ElementTree.fromstring(xmltree)]
while stack:
    elem = stack.pop()
    for child in list(elem):
        stack.append(child)
