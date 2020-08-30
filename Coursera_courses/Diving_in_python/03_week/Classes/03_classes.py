class Human:

    def __init__(self, name, age=0):
        self._name = name
        self._age = age

    def _say(self, text):
        print(text)

    def say_name(self):
        self._say(f'Hello, I am {self._name}')

    def say_how_old(self):
        self._say(f'I am {self._age} years old')


class Planet:

    def __init__(self, name, population=None):
        self.name = name
        self.population = population or []

    def add_human(self, human):
        print(f'Welcome to {self.name}, {human._name}!')
        self.population.append(human)

mars = Planet('Mars')

bob = Human('Bob', age=29)

mars.add_human(bob)

bob.say_name()
bob.say_how_old()
