class ToyClass:
    def instancemethod(self):
        return 'instance method called', self
    
    @classmethod
    def classmethod(cls):
        return 'class method called', cls

    @staticmethod
    def staticmethod():
        return 'static method called'


obj = ToyClass()
print(obj.instancemethod())
# ('instance method called', ToyClass instance at 0x10f47e7a0>)
print(ToyClass.instancemethod(obj))
# ('instance method called', ToyClass instance at 0x10f47e7a0>)

print(obj.classmethod())
# ('class method called', <class  ToyClass at 0x10f453a10>)

print(obj.staticmethod())
# static method called

print(ToyClass.classmethod())
# ('class method called', <class ToyClass at 0x10f453a10>)
print(ToyClass.staticmethod())
# 'static method called'
print(ToyClass.instancemethod())
# TypeError: unbound method instancemethod() 
# must be called with ToyClass instance as 
# first argument (got nothing instead)
