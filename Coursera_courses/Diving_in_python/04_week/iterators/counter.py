class Counter:
    """I count. That is all."""
    def __init__(self, initial=0):
        self.value = initial

    def increment(self):
        self.value += 1

    def get(self):
        return self.value


c = Counter(42)
c.increment()
print(c.get())
