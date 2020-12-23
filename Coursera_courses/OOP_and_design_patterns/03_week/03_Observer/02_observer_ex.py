from abc import ABC, abstractmethod
import pdb


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
    def __init__(self):
        self.ach = set()

    def update(self, message):
        self.ach.add(message['title'])
        print(self.ach)


class MessagePrinter(AbstractObserver):
    def __init__(self):
        self.achievements = []
        self.achv = set()

    def update(self, message):
        if self.achievements == []:
            self.achievements.append(message)
            self.achv.add(message['title'])
        else:
            if message['title'] not in self.achv:
                self.achv.add(message['title'])
                self.achievements.append(message)
        print(self.achievements)


# pdb.set_trace()
notifier = MessageNotifier()
printer1 = MessagePrinter()
# printer2 = MessagePrinter()

manager = NotificationManager()

# manager.subscribe(notifier)
manager.subscribe(printer1)
# manager.subscribe(printer2)

manager.notify({'text': 'Пройти основной сюжет игры', 'title': 'Воитель'})
manager.notify({'text': 'Пройти', 'title': 'Воитель'})
manager.notify({'text': 'Пройти', 'title': 'Воин'})
manager.notify({'text': 'Пройти основной сюжет игры', 'title': 'Воитель'})
manager.notify({'text': 'Пройти основной сюжет игры', 'title': 'Воитель'})
manager.notify({'text': 'Убить', 'title': 'Убийца_1'})
manager.notify({'text': 'Убить', 'title': 'Убийца_2'})
manager.notify({'text': 'Убить 100 противников', 'title': 'Убийца'})
manager.notify({'text': 'Убить 100 противников', 'title': 'Убийца'})
manager.notify({'text': 'Убить 100 противников', 'title': 'Убийца'})
manager.notify({'text': 'Убить 100 противников', 'title': 'Убийца'})
manager.notify({'text': 'Убить 100 противников', 'title': 'Убийца'})
