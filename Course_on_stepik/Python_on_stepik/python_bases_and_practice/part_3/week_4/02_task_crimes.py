import csv

with open('Crimes.csv') as f:
    reader = csv.DictReader(f)
    arr = []
    for row in reader:
        arr.append(row['Primary Type'])
    arr_count = {i: arr.count(i) for i in arr}
    print(arr_count)
