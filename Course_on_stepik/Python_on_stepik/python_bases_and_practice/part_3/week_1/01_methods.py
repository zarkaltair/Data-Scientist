print('abc' in 'abcda')
print('abce' in 'abcda')

print('cabcd'.find('abc'))

print(str.find.__doc__)

print('cabc'.index('abc'))
# print('cabc'.index('abcd'))

s = 'The man in black fled across the desert'
print(s.startswith('The man'))
print(s.startswith(('The man', 'The woman', 'The dog')))
print(str.startswith.__doc__)

s = 'image.png'
print(s.endswith('.png'))

s = 'abracadabra'
print(s.count('abra'))
print(str.count.__doc__)

s = 'abacaba'
print(s.find('aba'))
print(s.rfind('aba'))