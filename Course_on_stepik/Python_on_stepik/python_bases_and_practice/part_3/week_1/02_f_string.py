name = "Дмитрий"
age = 25
print(f"Меня зовут {name} Мне {age} лет.")
# Меня зовут Дмитрий. Мне 25 лет.

from math import pi

print(f"Значение числа pi: {pi:.2f}")
# Значение числа pi: 3.14

from datetime import datetime as dt
now = dt.now()
print(f"Текущее время {now:%d.%m.%Y %H:%M}")
# Текущее время 24.02.2017 15:51

x = 10
y = 5
print(f"{x} x {y} / 2 = {x * y / 2}")
# 10 x 5 / 2 = 25.0

planets = ["Меркурий", "Венера", "Земля", "Марс"]
print(f"Мы живим не планете {planets[2]}")
# Мы живим не планете Земля

planet = {"name": "Земля", "radius": 6378000}
print(f"Планета {planet['name']}. Радиус {planet['radius']/1000} км.")
# Планета Земля. Радиус 6378.0 км.

digits = {0: 'ноль', 'one': 'один'}
print(f"0 - {digits[0]}, 1 - {digits['one']}")
# 0 - ноль, 1 - один

name = "Дмитрий"
print(f"Имя: {name.upper()}")
# Имя: ДМИТИРИЙ

print(f"13 / 3 = {round(13/3)}")
# 13 / 3 = 4
