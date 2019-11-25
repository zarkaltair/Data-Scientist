import requests

res = requests.get('https://docs.python.org/3.8')
print(res.status_code)
print(res.headers['Content-Type'])

print(res.content)
print(res.text)

res = requests.get('https://docs.python.org/3.8/_static/py.png')
print(res.content)

with open('python.png', 'wb') as file:
    file.write(res.content)