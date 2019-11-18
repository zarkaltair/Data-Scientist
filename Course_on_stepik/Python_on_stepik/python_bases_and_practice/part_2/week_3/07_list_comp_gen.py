# list comprehension
arr = [(x, y) for x in range(3) for y in range(3) if y >= x]
print(arr)

# generator
gen = ((x, y) for x in range(3) for y in range(3) if y >= x)

# type object is generator
print(type(gen))

# output next item of generator
print(next(gen))
print(next(gen))
print(next(gen))