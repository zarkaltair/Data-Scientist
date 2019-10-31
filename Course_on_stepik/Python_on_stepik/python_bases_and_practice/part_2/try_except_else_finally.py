def divide(x, y):
    try:
        result = x / y
    except Exception as e:
        print(f'error {e}')
    else:
        print('result is ', result)
    finally:
        print('finally')

divide(2, 1)
divide(2, 0)
divide(2, [])