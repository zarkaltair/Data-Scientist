from contracts import contract


# preconditions
@contract(x='int,>=0')
def f(x):
    pass

'''
>>> f(-2)

ContractNotRespected: Breach for argument 'x' to f().
Condition -2 >= 0 not respected
checking: >=0       for value: Instance of <class 'int'>: -2 
checking: int,>=0   for value: Instance of <class 'int'>: -2

>>> f("Hello")

ContractNotRespected: Breach for argument 'x' to f().
Could not satisfy any of the 3 clauses in Int|np_scalar_int|np_scalar,array(int).
'''


# postconditions
@contract(returns='int,>=0')
def f(x):
    return x

'''
>>> f(-1)

ContractNotRespected: Breach for return value of f().
Condition -1 >= 0 not respected
checking: >=0       for value: Instance of <class 'int'>: -1   
checking: int,>=0   for value: Instance of <class 'int'>: -1   
Variables bound in inner context:
'''
