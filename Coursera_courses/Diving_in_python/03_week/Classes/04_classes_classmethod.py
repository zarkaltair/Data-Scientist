def extract_description(user_string):
    return 'open football championship'


def extract_date(user_string):
    return date(2020, 8, 30)


class Event:

    def __init__(self, description, event_date):
        self.description = description
        self.date = event_date

    def __str__(self):
        return f'Event \"{self.description}" at {self.date}'

    @classmethod
    def from_string(cls, user_input):
        description = extract_description(user_input)
        date = extract_date(user_input)
        return cls(description, date)


from datetime import date

event_description = 'Tell, what is @classmethod'
event_date = date.today()

event = Event(event_description, event_date)
print(event)

event = Event.from_string('add to my calendar opening football championship on 30 august 2020')
print(event)

print(dict.fromkeys('12345'))
