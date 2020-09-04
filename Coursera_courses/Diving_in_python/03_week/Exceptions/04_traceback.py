import traceback


try:
    with open('/file/not/found') as f:
        content = f.read()
except OSError as err:
    trace = traceback.print_exc()
    print(trace)
