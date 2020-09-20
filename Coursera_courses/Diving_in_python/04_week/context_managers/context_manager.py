class open_file:
    def __init__(self, filename, mode):
        self.f = open(filename, mode)

    def __enter__(self):
        return self.f

    def __exit__(self, *args):
        self.f.close()


with open_file('test.log', 'w') as f:
    f.write('Inside `open_file` context manager')


class suppress_exception:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == self.exc_type:
            print('Nothing happend.')
            return True


with suppress_exception(ZeroDivisionError):
    really_big_number = 1 / 0


import contextlib


with contextlib.suppress(ValueError):
    raise ValueError
