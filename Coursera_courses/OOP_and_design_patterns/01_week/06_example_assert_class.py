class DateType:
    def __init__(self, year=2000, month=1, day=1):
        assert year >= 0 and year < 3000
        assert month >= 1 and month <= 12 and day >= 1
        assert day <= [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1]
        self.year = year
        self.month = month
        self.day = day
