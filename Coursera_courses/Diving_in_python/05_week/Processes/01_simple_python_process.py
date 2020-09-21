import os
import time


pid = os.getpid()

while True:
    print(pid, time.time())
    time.sleep(2)

# ps axu | head -1; ps  axu | grep 01_simple_python_process.py
# sudo strace -p 58871
# lsof -p 58871
