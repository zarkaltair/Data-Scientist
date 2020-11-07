class A:
    value = 13
    
    def some_method(self):
        print(f"Method in A, value = {self.value}")
        
        
class B(A):
    
    def some_method(self):
        print(f"Method in B, value = {self.value}")
        

class C(B):
    value = 6
    
    def some_method(self):
        print(f"Method in C, value = {self.value}")

        
A().some_method()
B().some_method()
C().some_method()
print()
