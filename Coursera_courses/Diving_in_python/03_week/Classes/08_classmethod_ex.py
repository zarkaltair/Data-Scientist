from datetime import date


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_birth_year(cls, name, year):
        return cls(name, date.today().year - year)

    @staticmethod
    def is_adult(age):
        return age > 18


person1 = Person('Sarah', 25)
person2 = Person.from_birth_year('Roark', 1994)
print(person1.name, person1.age)
# Sarah 25
print(person2.name, person2.age)
# Roark 24
print(Person.is_adult(25))
# True
