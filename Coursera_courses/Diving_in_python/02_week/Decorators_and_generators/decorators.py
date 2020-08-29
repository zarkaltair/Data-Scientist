def decorator(func):
    return func


@decorator
def decorated():
    print('Hello!')


decorated = decorator(decorated)



def decorator(func):
    def new_func():
        pass
    return new_func

@decorator
def decorated():
    print('Hello!')

decorated()

print(decorated.__name__)



import functools

def logger(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        with open('log.txt', 'w') as f:
            f.write(str(result))

        return result
    return wrapped


@logger
def summator(num_list):
    return sum(num_list)

print(summator([1, 2, 3, 4, 5]))

print(summator.__name__)



def logger(filename):
    def decorator(func):
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            with open(filename, 'w') as f:
                f.write(str(result))
            return result
        return wrapped
    return decorator


@logger('new_log.txt')
def summator(num_list):
    return sum(num_list)

print(summator([1, 2, 3, 4, 5, 6]))
