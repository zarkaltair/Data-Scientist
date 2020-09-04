# attribute args
import os.path


filename = '/file/not/found'
try:
    if not os.path.exists(filename):
        raise ValueError('file not exists', filename)
except ValueError as err:
    message, filename = err.args[0], err.args[1]
    print(message, filename)
