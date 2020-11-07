# Переопределим метод в C
class A:
    value = 13
    
    def some_method(self):
        print(f"Method in A, value = {self.value}")
        
        
class B(A):
    pass


class C(A):
    
    def some_method(self):
        print(f"Method in С, value = {self.value}")

class D(B, C):
    pass

# Рассмотрим реализацию в D
D().some_method()
print()
