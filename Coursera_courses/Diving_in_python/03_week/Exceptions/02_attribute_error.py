try:
    with open('/file/not/found') as f:
        content = f.read()

except OSError as err:
    print(err.errno, err.strerror)
