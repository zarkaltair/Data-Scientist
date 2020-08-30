class Planet:
    """This class describes planets"""
    count = 0

    def __init__(self, name, population=None):
        self.name = name
        self.population = population or []
        Planet.count += 1

earth = Planet('Earth')
earth.mass = 5.97e24
mars = Planet('Mars')

print(Planet.count)
print(earth.__dict__)
print(Planet.__dict__)
print(dir(earth))
print(earth.__class__)


class Human:

    def __del__(self):
        print('Goodbye!')

human = Human()
del human
