import math


class Vec2d:

    def __init__(self, x, y):
        if type(x) is tuple:
            self.x = y[0] - x[0]
            self.y = y[1] - x[1]
        else:
            self.x = x
            self.y = y

    def __sub__(self, v):  # разность двух векторов
        return Vec2d(self.x - v.x, self.y - v.y)

    def __add__(self, v):  # сумма двух векторов
        return Vec2d(self.x + v.x, self.y + v.y)

    def __radd__(self, v):  # прибавить векторов
        self.x += v.x
        self.y += v.y

    def __len__(self):  # длинна вектора
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __mul__(self, z):
        if type(z) is Vec2d:  # скалярное умножение вектора на число
            return self.x * z.x + self.y * z.y
        else:  # умножение вектора на число
            return Vec2d(z * self.x, z * self.y)

    def __rmul__(self, z):
        return Vec2d(z * self.x, z * self.y)

    def int_pair(self):
        return int(self.x), int(self.y)

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
