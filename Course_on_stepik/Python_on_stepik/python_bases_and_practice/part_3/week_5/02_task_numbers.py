import sys
import requests


with open('dataset_24476_3.txt', 'r') as file:
    for line in file:
        api_url = 'http://numbersapi.com/' + line.strip() + '/math?json=true'
        res = requests.get(api_url)
        ans = res.json()
        if ans['found']:
            print('Interesting')
        else:
            print('Boring')