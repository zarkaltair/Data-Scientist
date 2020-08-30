class Robot:

    def __init__(self, power):
        self._power = power

    power = property()

    @power.setter
    def power(self, value):
        if value < 0:
            self._power = 0
        else:
            self._power = value

    @power.getter
    def power(self):
        return self._power

    @power.deleter
    def power(self):
        print('make robot useless')
        del self._power

wall_e = Robot(100)
wall_e.power = -20
print(wall_e.power)

del wall_e.power



class Robot:

    def __init__(self, power):
        self._power = power

    @property
    def power(self):
        # here may be other powerfull computes
        return self._power


wall_e = Robot(100)
print(wall_e.power)
