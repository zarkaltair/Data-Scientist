try:
    x = [1, 3, 'hello', 5]
    x.sort()
    print(x)
except TypeError:
    print('TypeError :(')

print('I can catch')


def f(x, y):
    try:
        return x / y
    except TypeError:
        print('TypeError')
    except ZeroDivisionError:
        print('Zero division :(')

print(f(5, 0))

def f(x, y):
    try:
        return x / y
    except (TypeError, ZeroDivisionError):
        print('Error :(')

print(f(5, 0))

def f(x, y):
    try:
        return x / y
    except (TypeError, ZeroDivisionError) as e:
        print(type(e))
        print(e)
        print(e.args)

print(f(5, 0))

def f(x, y):
    try:
        return x / y
    except:
        print('Error :(')

print(f(5, 0))

print(ZeroDivisionError.mro())


try:
    15 / 0
    # e
except ArithmeticError: # isinstance(e, ArithmeticError) == True
    print('arthmetic error')
except ZeroDivisionError:
    print('zero division')