import re

# print(re.match)
# print(re.search)
# print(re.findall)
# print(re.sub)


# pattern = r'a[abc]c'
# pattern = r'a[\w.]c'
pattern = r'a.c'
string = 'acc'
match_object = re.match(pattern, string)
# print(match_object)

string = 'abc, a.c, a,c, abc, aBc, azc, '
all_inclusions = re.findall(pattern, string)
print(all_inclusions)

fixed_typos = re.sub(pattern, 'abc', string)
print(fixed_typos)

# . ^ $ * + ? { } [ ] \ | ( ) — метасимволы

# [ ] — можно указать множество подходящих символов

# ^ - карет, обозначает либо начало строки, либо инвертирование группы символов. (например: "^[^0-9]" — не-цифра в начале строки).

# \d ~ [0-9] — цифры

# \D ~ [^0-9] — всё кроме цифр

# \s ~ [ \t\n\r\f\v] — пробельные символы

# \S ~ [^ \t\n\r\f\v] — всё кроме \s

# \w ~ [a-zA-Z0-9_] — буквы + цифры + _

# \W ~ [^a-zA-Z0-9_] — всё кроме того что в квадратных скобках
