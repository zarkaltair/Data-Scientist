import os

arr = []
for current_dir, dirs, files in os.walk('main'):
    for file in files:
        if '.py' in file:
            arr.append(current_dir)

arr = sorted(set(arr))
with open('ans.txt', 'w') as file:
    ans = '\n'.join(arr)
    file.write(ans)