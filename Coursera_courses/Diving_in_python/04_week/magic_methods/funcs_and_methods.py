class Class:
    def method(self):
        pass


obj = Class()

print(obj.method)
print(Class.method)



class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @property
    def full_name(self):
        return f'{self.first_name} {self.last_name}'


amy = User('Amy', 'Jones')

print(amy.full_name)
print(User.full_name)



class Property:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, obj, obj_type=None):
        if obj is None:
            return self

        return self.getter(obj)



class Class:
    @property
    def original(self):
        return 'original'

    @Property
    def custom_sugar(self):
        return 'custom sugar'

    def custom_pure(self):
        return 'custom pure'

    custom_pure = Property(custom_pure)


obj = Class()

print(obj.original)
print(obj.custom_sugar)
print(obj.custom_pure)
