# Переопределим метод в D
class A:
    value = 13
    
    def some_method(self):
        print(f"Method in A, value = {self.value}")
        
        
class B(A):
    pass


class C(A):
    pass


class D(B, C):
    
    def some_method(self):
        print(f"Method in D, value = {self.value}")

# Рассмотрим реализацию в D
D().some_method()
print()
