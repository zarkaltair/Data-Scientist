class DSU:
    def __init__(self, lst):
        self.values = lst
        self.len = len(lst)
        self.parents = list(range(self.len))

    def find_set(self, x):
        if self.parents[x] == x:
            return x
        self.parents[x] = self.find_set(self.parents[x])
        return self.parents[x]

    def is_joined(self,  a, b):
        if self.find_set(a) == self.find_set(b):
            return True
        else:
            return False

    def join(self, a, b):
        if not self.is_joined(a, b):
            self.parents[a] = b
