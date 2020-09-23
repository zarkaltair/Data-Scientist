def grep(pattern):
    print('start grep')
    while True:
        line = yield
        if pattern in line:
            print(line)

def grep_python_coroutine():
    g = grep('python')
    next(g)
    g.send('python is the best!')
    g.close()


g = grep_python_coroutine() # is g coroutione?
g
