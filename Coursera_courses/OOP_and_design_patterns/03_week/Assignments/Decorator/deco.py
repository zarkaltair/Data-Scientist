from inspect import isabstract
from abc import ABC, abstractmethod
# from test_decorator import Hero


class Hero:
    def __init__(self):
        self.positive_effects = []
        self.negative_effects = []

        self.stats = {
            "HP": 128,
            "MP": 42,
            "SP": 100,

            "Strength": 15,
            "Perception": 4,
            "Endurance": 8,
            "Charisma": 2,
            "Intelligence": 3,
            "Agility": 8,
            "Luck": 1
        }

    def get_positive_effects(self):
        return self.positive_effects.copy()

    def get_negative_effects(self):
        return self.negative_effects.copy()

    def get_stats(self):
        return self.stats.copy()


class AbstractEffect(Hero, ABC):

    @abstractmethod
    def get_stats():
        pass


class AbstractPositive(AbstractEffect):
    
    def __init__(self, base):
        self.base = base

    def get_stats(self):
        return self.base.get_stats()

    @abstractmethod
    def get_positive_effects():
        pass


class AbstractNegative(AbstractEffect):
    
    def __init__(self, base):
        self.base = base

    def get_stats(self):
        return self.base.get_stats()

    @abstractmethod
    def get_negative_effects():
        pass


class Berserk(AbstractPositive):
    def __init__(self, base):
        self.base = base
        self.stats = self.base.stats
        self.positive_effects = self.base.positive_effects
        self.negative_effects = self.base.negative_effects
        print(self.base.positive_effects, '--------------')
        self.base.stats['Strength'] += 7
        self.base.stats['Endurance'] += 7
        self.base.stats['Agility'] += 7
        self.base.stats['Luck'] += 7
        
        self.base.stats['Perception'] -= 3
        self.base.stats['Charisma'] -= 3
        self.base.stats['Intelligence'] -= 3

        self.base.stats['HP'] += 50

    def get_positive_effects(self):
        self.positive_effects.append('Berserk')
        return self.positive_effects.copy()


# class Blessing(AbstractPositive):
#     def __init__(self, base):
#         self.base = base
#         self.stats = self.base.stats.copy()

#         self.base.stats['Strength'] += 2
#         self.base.stats['Perception'] += 2
#         self.base.stats['Endurance'] += 2
#         self.base.stats['Charisma'] += 2
#         self.base.stats['Intelligence'] += 2
#         self.base.stats['Agility'] += 2
#         self.base.stats['Luck'] += 2

#     def get_positive_effects(self):
#         self.positive_effects = self.base.positive_effects.copy()
#         self.positive_effects.append('Blessing')
    
#     # def get_negative_effects(self):
#     #     self.negative_effects = self.base.negative_effects.copy()


# class Weakness(AbstractNegative):
#     def __init__(self, base):
#         self.base = base
#         self.stats = self.base.stats.copy()

#         self.base.stats['Strength'] -= 4
#         self.base.stats['Endurance'] -= 4
#         self.base.stats['Agility'] -= 4

#     # def get_positive_effects(self):
#     #     self.positive_effects = self.base.positive_effects.copy()
    
#     def get_negative_effects(self):
#         self.negative_effects = self.base.negative_effects.copy()
#         self.negative_effects.append('Weakness')


class Curse(AbstractNegative):
    def __init__(self, base):
        self.base = base
        self.stats = self.base.stats
        self.positive_effects = self.base.positive_effects
        self.negative_effects = self.base.negative_effects

        self.base.stats['Strength'] -= 2
        self.base.stats['Perception'] -= 2
        self.base.stats['Endurance'] -= 2
        self.base.stats['Charisma'] -= 2
        self.base.stats['Intelligence'] -= 2
        self.base.stats['Agility'] -= 2
        self.base.stats['Luck'] -= 2

    def get_positive_effects(self):
        return self.positive_effects.copy()
    
    def get_negative_effects(self):
        self.negative_effects.append('Curse')
        return self.negative_effects.copy()


# class EvilEye(AbstractNegative):
#     def __init__(self, base):
#         self.base = base
#         self.stats = self.base.stats.copy()
#         # self.positive_effects = self.base.positive_effects.copy()
#         # self.negative_effects = self.base.negative_effects.copy()

#         self.base.stats['Luck'] -= 10

#     # def get_positive_effects(self):
#     #     self.positive_effects = self.base.positive_effects.copy()
    
#     def get_negative_effects(self):
#         self.negative_effects = self.base.negative_effects.copy()
#         self.negative_effects.append('EvilEye')


# print(isabstract(AbstractEffect))
# print(isabstract(AbstractPositive))
# print(isabstract(AbstractNegative))
# print(isabstract(Berserk))
# print(isabstract(Blessing))
# print(isabstract(EvilEye))
