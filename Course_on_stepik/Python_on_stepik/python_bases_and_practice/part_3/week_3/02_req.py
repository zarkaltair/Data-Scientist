import requests

res = requests.get('https://yandex.ru/search/',
                   params={
                        'text': 'Stepik',
                        'test': 'test1',
                        'name': 'Name With Spaces',
                        'list': ['test1', 'test2']
                        })
print(res.status_code)
print(res.headers['Content-Type'])
print(res.url)
# print(res.text)