with open('dataset_24465_4.txt', 'r') as file, open('file_write.txt', 'w') as w:
    arr = []
    for line in file:
        arr.append(line.strip())
    for i in arr[::-1]:
        w.write(i + '\n')