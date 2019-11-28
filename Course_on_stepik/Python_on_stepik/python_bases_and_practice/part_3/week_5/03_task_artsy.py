import json
import requests
from pprint import pprint

client_id = '1456c262cd715cb0017d'
client_secret = '459da8ef1e53cba8a88b8d0508b866ab'

res = requests.post('https://api.artsy.net/api/tokens/xapp_token',
                    data={
                        'client_id': client_id,
                        'client_secret': client_secret
                    })

# разбираем ответ сервера
j = json.loads(res.text)

# достаем токен
token = j["token"]

# создаем заголовок, содержащий наш токен
headers = {"X-Xapp-Token": token}
# инициируем запрос с заголовком

with open('dataset_24476_4.txt', 'r') as file:
    dct = {}
    for line in file:
        res = requests.get('https://api.artsy.net/api/artists/' + line.strip(), headers=headers)
        j = json.loads(res.text)
        name = j['sortable_name']
        birthday = j['birthday']
        dct[name] = birthday
    arr = sorted(dct.items(), key=lambda x: (x[1], x[0]))
    for i in arr:
        print(i[0])
