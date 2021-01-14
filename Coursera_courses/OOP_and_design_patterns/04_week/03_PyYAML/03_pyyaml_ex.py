# демонстрация загрузки yaml по второму варианту
# важное замечание версия PyYAML - 3.13
import yaml


# класс определяющий пользовательский тип данных
class ExampleClass(yaml.YAMLObject):  # <-- добавим родительский класс yaml.YAMLObject
    yaml_tag = '!example_class'  # <-- добавим тег

    @classmethod
    def from_yaml(cls, loader, node):  # <-- добавим метод класса from_yaml
        # получаем данные из yaml
        value = loader.construct_mapping(node)
        # необходимо выбрать из полученных данных необходимые
        # для создания экземпляра класса ExampleClass
        return ExampleClass(*value)

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'ExampleClass, value - {self.value}'


# yaml строка
document = """!example_class {7}"""
# выполняем загрузку
obj = yaml.load(document)
# выведем полученный объект, ожидаем строку
print(obj)
# ExampleClass, value - 7
