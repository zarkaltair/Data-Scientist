import os


f = open('data.txt')
foo = f.readline()

if os.fork() == 0:
    # child process
    foo = f.readline()
    print('child:', foo)
else:
    # parent process
    foo = f.readline()
    print('parent:', foo)
