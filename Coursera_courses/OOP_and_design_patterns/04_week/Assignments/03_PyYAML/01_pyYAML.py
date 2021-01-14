import yaml
import random

from abc import ABC
from pprint import pprint


class AbstractLevel(yaml.YAMLObject):

    @classmethod
    def from_yaml(cls, loader, node):

        def get_easy_lvl(loader, node):
            data = loader.construct_mapping(node)
            _map = EasyLevel.Map()
            obj = EasyLevel.Objects()
            return {'map': _map, 'obj': obj}

        def get_medium_lvl(loader, node):
            data = loader.construct_mapping(node)
            _map = MediumLevel.Map()
            obj = MediumLevel.Objects()
            return {'map': _map, 'obj': obj}

        def get_hard_lvl(loader, node):
            data = loader.construct_mapping(node)
            _map = HardLevel.Map()
            obj = HardLevel.Objects()
            return {'map': _map, 'obj': obj}

        loader.add_constructor("!easy_level", get_easy_lvl)
        loader.add_constructor("!medium_level", get_medium_lvl)
        loader.add_constructor("!hard_level", get_hard_lvl)
        return loader.construct_mapping(node)


    @classmethod
    def get_map(cls):
        return cls.Map()

    @classmethod
    def get_objects(cls):
        return cls.Objects()

    class Map(ABC):
        pass

    class Objects(ABC):
        pass


class EasyLevel(AbstractLevel):
    yaml_tag = '!easy_level'

    class Map:
        def __init__(self):
            self.Map = [[0 for _ in range(5)] for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    if i == 0 or j == 0 or i == 4 or j == 4:
                        self.Map[j][i] = -1  # граница карты
                    else:
                        self.Map[j][i] = random.randint(0, 2)  # случайная характеристика области

        def get_map(self):
            return self.Map

    class Objects:
        def __init__(self):
            self.objects = [('next_lvl', (2, 2))]
            self.config = {}

        def get_objects(self, _map):
            for obj_name in ['rat']:
                coord = (random.randint(1, 3), random.randint(1, 3))
                intersect = True
                while intersect:
                    intersect = False
                    for obj in self.objects:
                        if coord == obj[1]:
                            intersect = True
                            coord = (random.randint(1, 3), random.randint(1, 3))

                self.objects.append((obj_name, coord))

            return self.objects


class MediumLevel(AbstractLevel):
    yaml_tag = '!medium_level'

    class Map:
        def __init__(self):
            self.Map = [[0 for _ in range(8)] for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    if i == 0 or j == 0 or i == 7 or j == 7:
                        self.Map[j][i] = -1  # граница карты
                    else:
                        self.Map[j][i] = random.randint(0, 2)  # случайная характеристика области

        def get_map(self):
            return self.Map

    class Objects:
        def __init__(self):
            self.objects = [('next_lvl', (4, 4))]
            self.config = {'enemy': []}

        def get_objects(self, _map):
            for obj_name in self.config['enemy']:
                coord = (random.randint(1, 6), random.randint(1, 6))
                intersect = True
                while intersect:
                    intersect = False
                    for obj in self.objects:
                        if coord == obj[1]:
                            intersect = True
                            coord = (random.randint(1, 6), random.randint(1, 6))

                self.objects.append((obj_name, coord))

            return self.objects


class HardLevel(AbstractLevel):
    yaml_tag = '!hard_level'

    class Map:
        def __init__(self):
            self.Map = [[0 for _ in range(10)] for _ in range(10)]
            for i in range(10):
                for j in range(10):
                    if i == 0 or j == 0 or i == 9 or j == 9:
                        self.Map[j][i] = -1  # граница карты :: непроходимый участок карты
                    else:
                        self.Map[j][i] = random.randint(-1, 8)  # случайная характеристика области

        def get_map(self):
            return self.Map

    class Objects:
        def __init__(self):
            self.objects = [('next_lvl', (5, 5))]
            self.config = {'enemy_count': 5, 'enemy': []}

        def get_objects(self, _map):
            for obj_name in self.config['enemy']:
                for tmp_int in range(self.config['enemy_count']):
                    coord = (random.randint(1, 8), random.randint(1, 8))
                    intersect = True
                    while intersect:
                        intersect = False
                        if _map[coord[0]][coord[1]] == -1:
                            intersect = True
                            coord = (random.randint(1, 8), random.randint(1, 8))
                            continue
                        for obj in self.objects:
                            if coord == obj[1]:
                                intersect = True
                                coord = (random.randint(1, 8), random.randint(1, 8))

                    self.objects.append((obj_name, coord))

            return self.objects


doc = '''!easy_level {}'''
level = yaml.load(doc)
# pprint(level)


# Levels = yaml.load(
# '''
# levels:
#     - !easy_level {}
#     - !medium_level
#         enemy: ['rat']
#     - !hard_level
#         enemy:
#             - rat
#             - snake
#             - dragon
#         enemy_count: 10
# ''')
# pprint(Levels)


# {'map': <main.EasyLevel.Map object at 0x7f27f0c297f0>, 
# 'obj': <main.EasyLevel.Objects object at 0x7f27ef5f1518>}


# Levels = {'levels': []}
# _map = EasyLevel.Map()
# _obj = EasyLevel.Objects()
# Levels['levels'].append({'map': _map, 'obj': _obj})

# _map = MediumLevel.Map()
# _obj = MediumLevel.Objects()
# _obj.config = {'enemy': ['rat']}
# Levels['levels'].append({'map': _map, 'obj': _obj})

# _map = HardLevel.Map()
# _obj = HardLevel.Objects()
# _obj.config = {'enemy': ['rat', 'snake', 'dragon'], 'enemy_count': 10}
# Levels['levels'].append({'map': _map, 'obj': _obj})
# pprint(Levels)

# medium_level = Levels['levels'][1]
# print(medium_level)
# print(medium_level['obj'].config)
# medium_map = medium_level['map'].get_map()
# pprint(medium_map)
# medium_obj = medium_level['obj'].get_objects(medium_map)
# pprint(medium_obj)
