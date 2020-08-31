class Pet:
    def __init__(self, name=None):
        self.name = name


class Dog(Pet):
    def __init__(self, name, breed=None):
        super().__init__(name)
        self.__breed = breed

    def say(self):
        return f'{self.name}: wow!'

    def get_breed(self):
        return self.__breed


dog = Dog('Sharik', 'Rex')
print(dog.name)
print(dog.say())

import json


class ExportJSON:
    def to_json(self):
        return json.dumps({
            'name': self.name,
            'breed': self.breed
            })


class ExDog(Dog, ExportJSON):
    def __init__(self, name, breed=None):
        super().__init__(name, breed)
        # super(ExDog, self).__init__(name)

    def get_breed(self):
        return f'breed: {self.name} - {self._Dog__breed}'


class WoolenDog(Dog, ExportJSON):
    def __init__(self, name, breed=None):
        super(Dog, self).__init__(name)
        self.breed = f'Dog breed is {breed}'


dog = ExDog('Belcka', breed='Laika')
# print(dog.to_json())


print(issubclass(int, object))
print(issubclass(Dog, object))
print(issubclass(Dog, Pet))
print(issubclass(Dog, int))

print(isinstance(dog, Dog))
print(isinstance(dog, Pet))
print(isinstance(dog, object))

print(ExDog.__mro__)

dog = WoolenDog('juchka', breed='Taksa')
print(dog.breed)

dog = ExDog('Focks', 'Mops')
print(dog.__dict__)
print(dog.get_breed())
