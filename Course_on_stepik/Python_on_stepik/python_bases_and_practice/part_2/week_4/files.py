f = open('test.txt', 'r')
# x = f.read(5)
# print(x)
# y = f.read()
# print(y)

x = f.read()
x = x.splitlines()
print(repr(x))

f.close()
print()


f = open('test.txt', 'r')
x = f.readline().strip()
print(repr(x))
x = f.readline()
print(repr(x))


f.close()
print()
f = open('test.txt', 'r')
for line in f:
    print(repr(line.strip()))
x = f.read()
print(repr(x))


f.close()