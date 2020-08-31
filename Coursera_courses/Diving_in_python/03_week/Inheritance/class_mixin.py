import json


class ExportJSON:
    def to_json(self):
        return json.dumps({
            'name': self.name,
            'breed': self.breed
            })


class ExDog(Dog, ExportJSON):
    pass

