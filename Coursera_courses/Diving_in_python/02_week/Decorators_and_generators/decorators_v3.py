def bold(func):
    def wrapped():
        return '<b>' + func() + '</b>'
    return wrapped


def italic(func):
    def wrapped():
        return '<i>' + func() + '</i>'
    return wrapped


@bold
@italic
def hello():
    return 'hello world'


# hello = bold(italic(hello))

print(hello())
