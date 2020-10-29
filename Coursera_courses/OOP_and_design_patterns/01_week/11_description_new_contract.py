from contracts import contract


# example #1
@new_contract
def even(x):
    if x % 2 != 0:
        msg = f'The number {x} is not even.'
        raise ValueError(msg)


@contract(x='int,even')
def foo(x):
    pass


# example #2
new_contract('short_list', 'list[N],N>0,N<=10')

@contract(a='short_list')
def bubble_sort(a):
    for bypass in range(len(a)-1):
        for i in range(len(a)-1-bypass):
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]


# linking the values ​​of various parameters
'''
В языке описания контрактов PyContracts используются переменные:

строчные латинские буквы — для любых объектов
заглавные латинские буквы — для целых чисел
'''
@contract(words='list[N](str),N>0',
          returns='list[N](>=0)')
def get_words_lengths(words):
    return [len(word) for word in words]
