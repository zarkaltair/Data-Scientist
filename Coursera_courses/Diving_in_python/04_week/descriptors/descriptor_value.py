class Value:
    def __init__(self):
        self.value = None

    @staticmethod
    def _prepare_value(value, comission):
        return value * comission

    def __get__(self, obj, obj_type):
        return self.value

    def __set__(self, obj, value):
        self.value = self._prepare_value(value, obj.comission)

class Class:
    attr = Value()

    def __init__(self, comission):
        self.comission = comission


instance = Class(0.1)
instance.attr = 10

print(instance.attr)
