import time


class timer():
    def __init__(self):
        self.start = time.time()

    def current_time(self):
        return time.time() - self.start
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(f'Elapsed: {self.current_time()}')


with timer() as t:
    time.sleep(1)
    print(f'Current: {t.current_time()}')
    time.sleep(1)
