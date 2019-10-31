'''
class Counter:
	def __init__(self):
		self.count = 0

	def inc(self):
		self.count += 1

	def reset(self):
		self.count = 0

print(Counter)
x = Counter()
x.inc()
print(x.count)
Counter.inc(x)
print(x.count)
x.reset()
print(x.count)
'''

class b:
    a = 0
    def __init__(self):
        self.a = 100
    def foo(self):
        self.a += 10
s = b()

#------атрибуты------
print(s.a) #100 - атрибут экземпляра равен 100
print(b.a) #0 - атрибут класса равен 0
b.a = 2 # меняем атрибут класса
print(s.a) #атрибут экземпляра по прежнему 100

#------методы экземпляров------
s.foo() #атрибут экземпляра + 10
print(s.a) #110 действительно, 110