# from inspect import isabstract
from abc import ABC, abstractmethod
# import pdb


class Hero:
    def __init__(self):
        self.positive_effects = []
        self.negative_effects = []

    def get_positive_effects(self):
        return self.positive_effects.copy()

    def get_negative_effects(self):
        return self.negative_effects.copy()


class AbstractEffect(Hero, ABC):
    def __init__(self, base):
        self.base = base

    @abstractmethod
    def get_positive_effects(self):
        return self.positive_effects

    @abstractmethod
    def get_negative_effects(self):
        return self.negative_effects


class AbstractPositive(AbstractEffect):
    def get_negative_effects(self):
        return self.base.get_negative_effects()


class AbstractNegative(AbstractEffect):
    def get_positive_effects(self):
        return self.base.get_positive_effects()


class Berserk(AbstractPositive):
    def get_positive_effects(self):
        positive = self.base.get_positive_effects()
        positive.append('Berserk')
        return positive.copy()


# class Berserk(AbstractPositive):
#     def get_positive_effects(self):
#         self.positive_effects = self.base.positive_effects.copy()
#         self.positive_effects.append('Berserk')
#         return self.positive_effects


class Curse(AbstractNegative):
    def get_negative_effects(self):
        negative = self.base.get_negative_effects()
        negative.append('Curse')
        return negative.copy()


# class Curse(AbstractNegative):
#     def get_negative_effects(self):
#         self.negative_effects = self.base.negative_effects.copy()
#         self.negative_effects.append('Curse')
#         return self.negative_effects


# pdb.set_trace()
'''
hero = Hero()
# print(hero.get_positive_effects())
# []
# print(hero.get_negative_effects())
# []
# print('---------------------')
# накладываем эффект
brs1 = Berserk(hero)
# print(brs1.get_positive_effects())
# ['Berserk']
# print(brs1.get_negative_effects())
# []
# print('---------------------')

# накладываем эффекты
brs2 = Berserk(brs1)
# print(brs2.get_positive_effects())
# ['Berserk', 'Berserk']
# print(brs2.get_negative_effects())
# []
# print('---------------------')

cur1 = Curse(brs2)
# print(cur1.get_positive_effects())
# ['Berserk', 'Berserk']
# print(cur1.get_negative_effects())
# ['Curse']
# print('---------------------')

brs3 = Berserk(cur1)
brs4 = Berserk(brs3)
print(brs4.get_positive_effects())
# print('---------------------')
print(brs4.get_negative_effects())
# print('---------------------')
# # снимаем эффект Berserk
# cur1.base = brs1
# print(cur1.get_positive_effects())
# # ['Berserk']
# print(cur1.get_negative_effects())
# # ['Curse']
print(brs4.base.base)
brs4.base.base = brs4.base.base.base
print(brs4.get_positive_effects())
print(brs4.get_negative_effects())

# print(isabstract(AbstractEffect))
# print(isabstract(AbstractPositive))
# print(isabstract(AbstractNegative))
# print(isabstract(Berserk))
# print(help(Berserk))
'''

print('---------------')
hero = Hero()
cur1 = Curse(hero)
cur2 = Curse(cur1)
cur3 = Curse(cur2)
print(cur3.get_positive_effects())
print(cur3.get_negative_effects())
brs1 = Berserk(cur3)
print(brs1.get_positive_effects())
print(brs1.get_negative_effects())
