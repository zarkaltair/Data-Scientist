class SampleClass:
    def __init__(self):
        self.public = 'public'
        self.__private = 'private'

    def public_method(self):
        print(f'private data: {self.__private}')


c = SampleClass()
print(c.public)
print(c.public_method())
# c.__private  # raise error
print(c._SampleClass__private)
