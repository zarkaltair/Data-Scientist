import re

pattern = r'ab*a'
pattern = r'ab+a'
pattern = r'ab?a'
pattern = r'ab{3}a'
pattern = r'ab{2,4}a'
string = 'aa, aba, abba, abbba, abbbba'
all_inclusions = re.findall(pattern, string)
print(all_inclusions)

pattern = r'a[ab]+a'
pattern = r'a[ab]+?a'
string = 'abaaba'
print(re.match(pattern, string))
print(re.findall(pattern, string))
