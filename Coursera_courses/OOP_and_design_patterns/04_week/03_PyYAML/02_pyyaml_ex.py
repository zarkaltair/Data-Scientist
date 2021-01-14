# демонстрация загрузки yaml по первому варианту
# важное замечание версия PyYAML - 3.13
import yaml


# класс определяющий пользовательский тип данных
class ExampleClass:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'ExampleClass, value - {self.value}'


# функция конструктор для типа данных ExampleClass
def constuctor_example_class(loader, node):
    # получаем данные из yaml
    value = loader.construct_mapping(node)
    # необходимо выбрать из полученных данных необходимые
    # для создания экземпляра класса ExampleClass
    return ExampleClass(*value)


# регистрируем конструктор
yaml.add_constructor('!example_class', constuctor_example_class)
# yaml строка
document = """!example_class {5}"""
# выполняем загрузку
obj = yaml.load(document)
# выведем полученный объект, ожидаем строку
print(obj)
# ExampleClass, value - 5
