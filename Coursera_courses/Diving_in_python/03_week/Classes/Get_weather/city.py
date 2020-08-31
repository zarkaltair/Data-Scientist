import requests

from pprint import pprint
from datetime import datetime


class OpenWeatherForecast:

    def __init__(self):
        self._city_cache = {}

    def get(self, city):
        if city in self._city_cache:
            return self._city_cache[city]
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid=19635a07009bb029f8569e0bad9be9d3'
        print('Sending HTTP request')
        data = requests.get(url).json()
        day_temp = data['main']['temp']
        date = datetime.fromtimestamp(data['dt'])
        forecast = [{'date': date, 'high_temp': day_temp}]
        self._city_cache[city] = forecast
        return forecast


class CityInfo:

    def __init__(self, city, weather_forecast=None):
        self.city = city
        self._weather_forecast = weather_forecast or OpenWeatherForecast()

    def weather_forecast(self):
        return self._weather_forecast.get(self.city)


def _main():
    weather_forecast = OpenWeatherForecast()
    for i in range(5):
        city_info = CityInfo('Moscow', weather_forecast=weather_forecast)
        forecast = city_info.weather_forecast()


if __name__ == '__main__':
    _main()
