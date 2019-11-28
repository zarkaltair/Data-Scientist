from pprint import pprint
import json

with open('04_test_01.json', 'r') as f:
    data = json.load(f)

def count(k):
    if dd[k] == []:
        dct[k] += []
    else:
        for i in dd[k]:
            # if k not in dct[i]:
            dct[i] += [k]
            count(i)

dd = {}
dct = {}
for i in data:
    dct[i['name']] = []
    dd[i['name']] = i['parents']

for k, v in dd.items():
    count(k)

# for i in sorted(dct):
#     print(i, ':', len(set(dct[i])) + 1)

# pprint(dct)
# pprint(dd)


def count(dct, key, val=None):
    if val is None:
        val = set()
    val.add(key)
    for i in dct[key] - val:
        count(dct, i, val)
    return val


with open('04_test_01.json', 'r') as f:
    data = json.load(f)

dct = {line['name']: [] for line in data}

for line in data:
    for key in dct:
        if key in line['parents']:
            dct[key].append(line['name'])

for key in dct:
    dct[key] = set(dct[key])

for key in sorted(dct.keys()):
    print(key, ':', len(count(dct, key)))
