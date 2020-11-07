from abc import ABC, abstractmethod


class A:
    @abstractmethod
    def do_something(self):
        print('Hi!')

# a = A()
# print(a.do_something())


class A(ABC):
    @abstractmethod
    def do_something(self):
        print('Hi!')


# a = A()

class B(A):
    def do_something(self):
        print('Hi2!')

    def do_something_else(self):
        print('Hello')


b = B()
print(b.do_something())
print(b.do_something_else())
