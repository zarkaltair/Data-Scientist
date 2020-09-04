import sys
import requests


url = sys.argv[1]

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

except requests.Timeout:
    print('error timeout, url: ', url)

except requests.HTTPError as err:
    code = err.response.status_code
    print(f'error url: {url}, code {code}')

except requests.RequestException:
    print('error download url: ', url)

else:
    print(requests.content)
