import simplecrypt


with open('encrypted.bin', 'rb') as inp:
    encrypted = inp.read()

with open('passwords.txt', 'r') as file:
    arr_pass = []
    for line in file:
        arr_pass.append(line.strip())

for i in arr_pass:
    try:
        decrypted = simplecrypt.decrypt(data=encrypted, password=i)
    except:
        print('Bad password')

# print(encrypted)
# print(arr_pass)
print(decrypted)