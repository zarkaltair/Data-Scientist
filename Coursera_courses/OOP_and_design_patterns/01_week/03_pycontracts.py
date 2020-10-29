from contracts import contract


@contract(a='int,>0',
          b='list[N],N>0',
          returns='list[N]')
def my_function(a, b):
    pass


@contract
def my_function(a: 'int,>0',
                b: 'list[N],N>0') -> 'list[N]':
    pass


@contract
def my_function(a, b):
    """Function description.
       :type a: int,>0
       :type b: list[N],N>0
       :rtype: list[N]
    """
    pass
