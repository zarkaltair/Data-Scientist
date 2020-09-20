class NonNegative:
    def __get__(self, instance, owner):
        return magically_get_value(...)

    def __set__(self, instance, value):
        assert value >= 0, 'non-negative value required'
        magically_set_value(...)

    def __delete__(self, instance):
        magically_delete_value(...)


class VerySafe:
    x = NonNegative()
    y = NonNegative()


very_safe = VerySafe()
very_safe.x = 42
print(very_safe.x)

# very_safe.x = -42