class MemorizingDict(dict):
    history = deque(maxlen=10)

    def set(self, key, value):
        self.history.append(key)
        self[key] = value

    def get_history(self):
        return self.history


d = MemorizingDict({'foo': 42})
d.set('baz', 100500)
print(d.get_history())

d = MemorizingDict()
d.set('boo', 500100)
print(d.get_history())
