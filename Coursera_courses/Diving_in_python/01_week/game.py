import random


number = random.randint(0, 101)

while True:
    answer = input('Enter number: ')
    if not answer or answer == 'exit':
        break
    
    if not answer.isdigit():
        print('Enter true number!')
        continue

    user_answer = int(answer)

    if user_answer > number:
        print('Number is less')
    elif user_answer < number:
        print('Number is better')
    else:
        print('Excellent!!!')
        break
