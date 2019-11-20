class BadName(Exception):
    pass


def greet(name):
    if name[0].isupper():
        return 'Hello, ' + name
    else:
        raise BadName(name + ' is inappropriate name')


# указанное имя переменной начинающееся с _ не будет импортировано из модуля
_GREETING = 'Hello '

print('Import is execution')

# указываем все имена которые нужно импортировать из модуля
__all__ = ['BadName', 'greet']