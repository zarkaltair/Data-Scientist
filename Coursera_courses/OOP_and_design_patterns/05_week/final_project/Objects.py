from abc import ABC, abstractmethod


class AbstractObject(ABC):

    @abstractmethod
    def draw(self, display):
        pass


class Interactive(ABC):

    @abstractmethod
    def interact(self, engine, hero):
        pass


class Ally(AbstractObject, Interactive):

    def __init__(self, icon, action, position):
        self.sprite = icon
        self.action = action
        self.position = position

    def interact(self, engine, hero):
        self.action(engine, hero)

    def draw(self, display):
        display.draw_object(self.sprite, self.position)


class Creature(AbstractObject):

    def __init__(self, icon, stats, position):
        self.sprite = icon
        self.stats = stats
        self.position = position
        self.calc_max_hp()
        self.hp = self.max_hp

    def calc_max_hp(self):
        self.max_hp = 5 + self.stats["endurance"] * 2


class Enemy(Creature, Interactive):

    def __init__(self, icon, stats, xp, position):
        self.sprite = icon
        self.stats = stats
        self.xp = xp
        self.position = position

    def interact(self, engine, hero):
        hero.hp -= self.stats['strength']
        hero.exp += self.stats['experience']
        hero.level_up()

    def draw(self, display):
        display.draw_object(self.sprite, self.position)


class Hero(Creature):

    def __init__(self, stats, icon):
        pos = [1, 1]
        self.level = 1
        self.exp = 0
        self.gold = 0
        super().__init__(icon, stats, pos)

    def level_up(self):
        while self.exp >= 100 * (2 ** (self.level - 1)):
            self.level += 1
            self.stats["strength"] += 2
            self.stats["endurance"] += 2
            self.calc_max_hp()
            self.hp = self.max_hp

    def draw(self, display):
        display.draw_object(self.sprite, self.position)


class Effect(Hero):

    def __init__(self, base):
        self.base = base
        self.stats = self.base.stats.copy()
        self.apply_effect()

    @property
    def position(self):
        return self.base.position

    @position.setter
    def position(self, value):
        self.base.position = value

    @property
    def level(self):
        return self.base.level

    @level.setter
    def level(self, value):
        self.base.level = value

    @property
    def gold(self):
        return self.base.gold

    @gold.setter
    def gold(self, value):
        self.base.gold = value

    @property
    def hp(self):
        return self.base.hp

    @hp.setter
    def hp(self, value):
        self.base.hp = value

    @property
    def max_hp(self):
        return self.base.max_hp

    @max_hp.setter
    def max_hp(self, value):
        self.base.max_hp = value

    @property
    def exp(self):
        return self.base.exp

    @exp.setter
    def exp(self, value):
        self.base.exp = value

    @property
    def sprite(self):
        return self.base.sprite

    @abstractmethod
    def apply_effect(self):
        pass


# FIXME
# add classes
class Berserk(Effect):
    def apply_effect(self):
        stats = self.base.stats
        stats['strength'] += 10
        stats['endurance'] += 10
        stats['intelligence'] -= 3
        stats['luck'] += 1
        return stats


class Blessing(Effect):
    def apply_effect(self):
        stats = self.base.stats
        stats['strength'] += 5
        stats['endurance'] += 5
        stats['intelligence'] += 5
        stats['luck'] += 5
        return stats


class Weakness(Effect):
    def apply_effect(self):
        stats = self.base.stats
        stats['strength'] -= 5
        stats['endurance'] -= 5
        stats['intelligence'] -= 5
        stats['luck'] -= 5
        return stats
