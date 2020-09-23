def MyRageGenerator(top):
    current = 0
    while current < top:
        yield current
        current += 1


counter = MyRageGenerator(3)
print(counter)

for it in counter:
    print(it)
