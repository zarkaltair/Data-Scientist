NewClass = type('NewClass', (), {})

print(NewClass)
print(NewClass())



class Mete(type):
    def __new__(cls, name, parents, attrs):
        print(f'Creating {name}')

        if 'class_id' not in attrs:
            attrs['class_id'] = name.lower()

        return super().__new__(cls, name, parents, attrs)


class A(metaclass=Mete):
    pass

print(f'A.class_id: "{A.class_id}"')



class Mete(type):
    def __init__(cls, name, bases, attrs):
        print(f'Initializing - {name}')

        if not hasattr(cls, 'registry'):
            cls.registry = {}
        else:
            cls.registry[name.lower()] = cls

        super().__init__(name, bases, attrs)


class Base(metaclass=Mete): pass

class A(Base): pass

class B(Base): pass

print(Base.registry)
print(Base.__subclasses__())
