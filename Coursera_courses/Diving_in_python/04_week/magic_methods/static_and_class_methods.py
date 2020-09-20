class StaticMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, obj_type=None):
        return self.func


class ClassMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, obj_type=None):
        if obj_type is None:
            obj_type = type(obj)

        def new_func(*args, **kwargs):
            return self.func(obj_type, *args, **kwargs)

        return new_func
