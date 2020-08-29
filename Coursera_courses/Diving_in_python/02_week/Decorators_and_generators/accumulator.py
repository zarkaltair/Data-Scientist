def accumulator():
    total = 0
    while True:
        value = yield total
        print(f'Got: {value}')

        if not value: break
        total += value

generator = accumulator()

print(next(generator))
print(f'Accumulated: {generator.send(1)}')
print(f'Accumulated: {generator.send(1)}')
print(f'Accumulated: {generator.send(1)}')
print(next(generator))
