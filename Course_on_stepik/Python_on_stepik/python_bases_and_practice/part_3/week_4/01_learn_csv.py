import csv

with open('01_example.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

print()
with open('01_example.tsv') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        print(row)

students = [
['Greg', 'Dean', 70, 80, 90, 'Good job, Greg'],
['Wirt', 'Wood', 80, 80.2, 80, 'Nicely done']
]

with open('01_example.csv', 'a') as f:
    writer = csv.writer(f)
    for student in students:
        writer.writerow(student)

with open('01_example.csv', 'a') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    # writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerows(students)