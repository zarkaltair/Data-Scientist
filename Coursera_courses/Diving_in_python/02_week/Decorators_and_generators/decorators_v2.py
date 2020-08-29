def first_decorator(func):
    def wrapped():
        print('Inside first_decorator product')
        return func()
    return wrapped


def second_decorator(func):
    def wrapped():
        print('Inside second_decorator product')
        return func()
    return wrapped


@first_decorator
@second_decorator
def decorated():
    print('Finally called...')


print(decorated())

# decorated = first_decorator(second_decorator(decorated))
