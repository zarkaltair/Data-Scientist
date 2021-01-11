from abc import ABC, abstractmethod


class HeroFactory(ABC):
    @abstractmethod
    def create_hero(self, name):
        pass

    @abstractmethod
    def create_weapon(self):
        pass

    @abstractmethod
    def create_spell(self):
        pass


class WarriorFactory(HeroFactory):
    def create_hero(self, name):
        return Warrior(name)

    def create_spell(self):
        return Power()

    def create_weapon(self):
        return Claymore()


class Warrior:
    def __init__(self, name):
        self.name = name
        self.spell = None
        self.weapon = None

    def add_weapon(self, weapon):
        self.weapon = weapon

    def add_spell(self, spell):
        self.spell = spell

    def hit(self):
        print(f'Warrior {self.name} hits with {self.weapon.hit()}')
        self.weapon.hit()

    def cast(self):
        print(f'Warrior {self.name} casts {self.spell.cast()}')
        self.spell.cast()


class Claymore:
    def hit(self):
        return 'Claymore'


class Power:
    def cast(self):
        return 'Power'


class MageFactory(HeroFactory):
    def create_hero(self, name):
        return Mage(name)
    
    def create_weapon(self):
        return Staff()
    
    def create_spell(self):
        return Fireball()


class Mage:
    def __init__(self, name):
        self.name = name
        self.weapon = None
        self.armor = None
        self.spell = None

    def add_weapon(self, weapon):
        self.weapon = weapon

    def add_spell(self, spell):
        self.spell = spell

    def hit(self):
        print(f"Mage {self.name} hits with {self.weapon.hit()}")
        self.weapon.hit()

    def cast(self):
        print(f"Mage {self.name} casts {self.spell.cast()}")
        self.spell.cast()


class Staff:
    def hit(self):
        return "Staff"


class Fireball:
    def cast(self):
        return "Fireball"


class AssassinFactory(HeroFactory):
    def create_hero(self, name):
        return Assassin(name)

    def create_weapon(self):
        return Dagger()

    def create_spell(self):
        return Invisibility()


class Assassin:
    def __init__(self, name):
        self.name = name
        self.weapon = None
        self.armor = None
        self.spell = None

    def add_weapon(self, weapon):
        self.weapon = weapon

    def add_spell(self, spell):
        self.spell = spell

    def hit(self):
        print(f'Assassin {self.name} hits with {self.weapon.hit()}')
        self.weapon.hit()

    def cast(self):
        print(f'Assassin {self.name} casts {self.spell.cast()}')
        self.spell.cast()


class Dagger:
    def hit(self):
        return 'Dagger'


class Invisibility:
    def cast(self):
        return 'Invisibility'


def create_hero(factory):
    hero = factory.create_hero('Nagibator')

    weapon = factory.create_weapon()
    spell = factory.create_spell()

    hero.add_weapon(weapon)
    hero.add_spell(spell)
    return hero


player = create_hero(MageFactory())
player.hit()
player.cast()

player = create_hero(AssassinFactory())
player.hit()
player.cast()
