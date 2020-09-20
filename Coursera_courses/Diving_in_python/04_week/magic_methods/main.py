class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def get_email_data(self):
        return {
            'name': self.name,
            'email': self.email
        }

jane = User('Jane Doe', 'janedoe@example.com')

print(jane.get_email_data())


class Singleton:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

a = Singleton()
b = Singleton()
print(a is b)


class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __str__(self):
        return f'{self.name} <{self.email}>'

jane = User('Jane Doe', 'janedoe@example.com')

print(jane)


class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __hash__(self):
        return hash(self.email)

    def __eq__(self, obj):
        return self.email == obj.email

jane = User('Jane Doe', 'jdoe@example.com')
joe = User('Joe Doe', 'jdoe@example.com')
print(jane == joe)

print(hash(jane))
print(hash(joe))

user_email_map = {user: user.name for user in [jane, joe]}
print(user_email_map)


class Researcher:
    def __getattr__(self, name):
        return 'Nothing found :()\n'

    def __getattribute__(self, name):
        print(f'Looking for {name}')
        return object.__getattribute__(self, name)


obj = Researcher()

print(obj.attr)
print(obj.method)
print(obj.DFJ2H3J00KLL)


class Ignorant:
    def __setattr__(self, name, value):
        print(f'Not gonna set {name}!')

obj = Ignorant()
obj.math = True
# print(obj.math)


class Polite:
    def __delattr__(self, name):
        value = getattr(self, name)
        print(f'Good {name}, you were {value}!')

        object.__delattr__(self, name)


obj = Polite()

obj.attr = 10
del obj.attr


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            with open(self.filename, 'a') as f:
                f.write('Oh Danny boy...')
            
            return func(*args, *kwargs)
        return wrapped


logger = Logger('log.txt')

@logger
def completely_useless_function():
    pass


import random


class NoisyInt:
    def __init__(self, value):
        self.value = value

    def __add__(self, obj):
        noise = random.uniform(-1, 1)
        return self.value + obj.value + noise


a = NoisyInt(10)
b = NoisyInt(20)

for _ in range(3):
    print(a + b)


class PascalList:
    def __init__(self, original_list=None):
        self.container = original_list or []

    def __getitem__(self, index):
        return self.container[index - 1]

    def __setitem__(self, index, value):
        self.container[index - 1] = value

    def __str__(self):
        return self.container.__str__()


numbers = PascalList([1, 2, 3, 4, 5])
print(numbers[1])

numbers[5] = 25
print(numbers)
