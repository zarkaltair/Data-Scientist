class SomeObject:
    def __init__(self):
        self.integer_field = 0
        self.float_field = 0.0
        self.string_field = ''


class NullHandler:
    def __init__(self, kind):
        self.kind = kind


class IntHandler():
    pass


class FloatHandler():
    pass


class StrHandler():
    pass


chain = IntHandler(FloatHandler(StrHandler(NullHandler)))

obj = SomeObject()
obj.integer_field = 42
obj.float_field = 3.14
obj.string_field = "some text"
chain = IntHandler(FloatHandler(StrHandler(NullHandler)))
chain.handle(obj, EventGet(int))
# 42
chain.handle(obj, EventGet(float))
# 3.14
chain.handle(obj, EventGet(str))
# 'some text'
chain.handle(obj, EventSet(100))
chain.handle(obj, EventGet(int))
# 100
chain.handle(obj, EventSet(0.5))
chain.handle(obj, EventGet(float))
# 0.5
chain.handle(obj, EventSet('new text'))
chain.handle(obj, EventGet(str))
# 'new text'
