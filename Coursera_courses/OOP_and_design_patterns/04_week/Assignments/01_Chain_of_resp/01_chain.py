class SomeObject:
    def __init__(self):
        self.integer_field = 0
        self.float_field = 0.0
        self.string_field = ''


class EventGet:
    def __init__(self, kind):
        self.kind = kind
        self.value = None


class EventSet:
    def __init__(self, value):
        self.kind = type(value)
        self.value = value


class NullHandler:
    def __init__(self, successor=None):
        self.__successor = successor

    def handle(self, _obj, event):
        if self.__successor is not None:
            return self.__successor.handle(_obj, event)


class IntHandler(NullHandler):
    def handle(self, _obj, event):
        if event.kind == int and event.value is not None:
            _obj.integer_field = event.value
            return _obj.integer_field
        elif event.kind == int and event.value is None:
            return _obj.integer_field
        else:
            return super().handle(_obj, event)


class FloatHandler(NullHandler):
    def handle(self, _obj, event):
        if event.kind == float and event.value is not None:
            _obj.float_field = event.value
            return _obj.float_field
        elif event.kind == float and event.value is None:
            return _obj.float_field
        else:
            return super().handle(_obj, event)


class StrHandler(NullHandler):
    def handle(self, _obj, event):
        if event.kind == str and event.value is not None:
            _obj.string_field = event.value
            return _obj.string_field
        elif event.kind == str and event.value is None:
            return _obj.string_field
        else:
            return super().handle(_obj, event)


'''
obj = SomeObject()
obj.integer_field = 42
obj.float_field = 3.14
obj.string_field = "some text"

chain = IntHandler(FloatHandler(StrHandler(NullHandler())))

print(chain.handle(obj, EventGet(int)))
# 42
print(chain.handle(obj, EventGet(float)))
# 3.14
print(chain.handle(obj, EventGet(str)))
# 'some text'

print(chain.handle(obj, EventSet(-100)))
print(chain.handle(obj, EventGet(int)))
# 100

print(chain.handle(obj, EventSet(0.5)))
print(chain.handle(obj, EventGet(float)))
# 0.5

print(chain.handle(obj, EventSet('new text')))
print(chain.handle(obj, EventGet(str)))
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
