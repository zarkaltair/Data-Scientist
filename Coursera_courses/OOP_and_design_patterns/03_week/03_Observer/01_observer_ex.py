from abc import ABC, abstractmethod


class NotificationManager:
    def __init__(self):
        self.__subscribers = set()

    def subscribe(self, subscriber):
        self.__subscribers.add(subscriber)

    def unsubscribe(self, subscriber):
        self.__subscribers.remove(subscriber)

    def notify(self, message):
        for subscriber in self.__subscribers:
            subscriber.update(message)


class AbstractObserver(ABC):
    @abstractmethod
    def update(self, message):
        pass


class MessageNotifier(AbstractObserver):
    def __init__(self, name):
        self.__name = name

    def update(self, message):
        print(f'{self.__name} recieved message!')


class MessagePrinter(AbstractObserver):
    def __init__(self, name):
        self.__name = name

    def update(self, message):
        print(f'{self.__name} recieved message: {message}')


notifier = MessageNotifier('Notifier1')
printer1 = MessagePrinter('Printer1')
printer2 = MessagePrinter('Printer2')

manager = NotificationManager()

manager.subscribe(notifier)
manager.subscribe(printer1)
manager.subscribe(printer2)

manager.notify('Hi!')
