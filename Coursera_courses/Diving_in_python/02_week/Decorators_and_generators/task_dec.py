import json


def to_json(func):
    return json.dumps(func)

@to_json
def get_data():
    return {'data': 42}

print(get_data())  # вернёт '{"data": 42}'
