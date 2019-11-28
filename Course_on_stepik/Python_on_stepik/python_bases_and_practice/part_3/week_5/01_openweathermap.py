from pprint import pprint
import requests

city = input('City? ')

api_key = 'e7bb679b81c3710fc4192bcc1018a907'
api_url = 'http://api.openweathermap.org/data/2.5/weather'

params = {
    'q': city,
    'appid': api_key,
    'units': 'metric'
}

'https://api.openweathermap.org/data/2.5/weather?q=Moscow&appid=e7bb679b81c3710fc4192bcc1018a907'

res = requests.get(api_url, params=params)
# print(res.status_code)
# print(res.headers['Content-Type'])
# pprint(res.json()) # return json.loads(res.text)

data = res.json()
temp = data['main']['temp']
print(f'Current temperature in {city} is {temp}')