import time

class Loggable:
    def log(self, msg):
        print(str(time.ctime()) + ": " + str(msg))


class LoggableList(list, Loggable):
    def append(self, var):
        x = super(LoggableList, self).append(var)
        Loggable.log(self, var)
        return x


lst = LoggableList([5, 3, 55])
# print(lst)
lst.append(5)
# print(lst)