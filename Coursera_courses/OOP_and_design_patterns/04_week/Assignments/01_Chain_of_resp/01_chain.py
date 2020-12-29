class SomeObject:
    def __init__(self):
        self.integer_field = 0
        self.float_field = 0.0
        self.string_field = ''


class EventGet:
    def __init__(self, kind):
        self.kind = kind


class EventSet():
    pass


class NullHandler:
    def __init__(self, successor=None):
        self.__successor = successor

    def handle(self, obj, event):
        if self.__successor is not None:
            return self.__successor.handle(obj, event)


class IntHandler(NullHandler):
    def handle(self, obj, event):
        if event.kind == int:
            return obj.integer_field
        else:
            return super().handle(obj, event)


class FloatHandler(NullHandler):
    def handle(self, obj, event):
        if event.kind == float:
            return obj.float_field
        else:
            return super().handle(obj, event)


class StrHandler(NullHandler):
    def handle(self, obj, event):
        if event.kind == str:
            return obj.string_field
        else:
            return super().handle(obj, event)


obj = SomeObject()
obj.integer_field = 42
obj.float_field = 3.14
obj.string_field = "some text"

chain = IntHandler(FloatHandler(StrHandler(NullHandler)))

print(chain.handle(obj, EventGet(int)))
# 42
print(chain.handle(obj, EventGet(float)))
# 3.14
print(chain.handle(obj, EventGet(str)))
# 'some text'
'''
chain.handle(obj, EventSet(100))
chain.handle(obj, EventGet(int))
# 100

chain.handle(obj, EventSet(0.5))
chain.handle(obj, EventGet(float))
# 0.5

chain.handle(obj, EventSet('new text'))
chain.handle(obj, EventGet(str))
# 'new text'


# предположим, что выше по коду у нас определены константы E_INT, E_FLOAT, E_STR
# выделим из конструкции, переданной в print, словарь и поместим его в переменную types
types = {int: E_INT, float: E_FLOAT, str: E_STR}
# видим, что ключами в словаре служат типы данных int, float, str. Такое возможно, потому что это хэшируемые объекты. 
# В этом вы можете убедиться сами:
hash(int)  # вернет значение хэша
hasattr(int, '__hash__')  # вернет True
# таким образом у нас получается конструкция
types[type(2)]
# это обычное обращение к словарю для получения значения по ключу. В данном примере ключом выступает, 
# возвращаемый из функции type результат
type(2)  # вернет int, поскольку 2 - целое число и объект типа int
# вся конструкция свелась к 
types[int]
# такой вызов вернет значение, лежащее в словаре по ключу int, а именно значение константы E_INT
'''
