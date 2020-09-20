class Descriptor:
    def __get__(self, obj, obj_type):
        print('get')

    def __set__(self, obj, value):
        print('set')

    def __delete__(self, obj):
        print('delete')


class Class:
    attr = Descriptor()


instance = Class()

# instance.attr
# instance.attr = 10
# del instance.attr


class Descr:
    def __get__(self, instance, owner):
        print(instance, owner)

    def __set__(self, instance, value):
        print(instance, value)


class A:
    attr = Descr()

class B(A):
    pass

# A.attr
# A().attr

# B.attr
# B().attr

instance = A()
instance.attr = 42
