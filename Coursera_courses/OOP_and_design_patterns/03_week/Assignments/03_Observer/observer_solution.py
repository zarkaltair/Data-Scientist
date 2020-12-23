from abc import ABC, abstractmethod


class ObservableEngine(Engine):
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


class ShortNotificationPrinter(AbstractObserver):
    def __init__(self):
        self.achievements = set()

    def update(self, message):
        self.achievements.add(message['title'])


class FullNotificationPrinter(AbstractObserver):
    def __init__(self):
        self.achievements = []
        self.achv_title = set()

    def update(self, message):
        if self.achievements == []:
            self.achv_title.add(message['title'])
            self.achievements.append(message)
        else:
            if message['title'] not in self.achv_title:
                self.achv_title.add(message['title'])
                self.achievements.append(message)


'''
AssertionError: Тест 14.3. Не верное сохранение достижений у подписчика FullNotificationPrinter после получения уведомлений. 
                Каждое достижение должно быть уникальным (то есть учтено только один раз).

assert 10 == 5
+  where 10 = len([{'text': 'Пройти основной сюжет игры', 'title': 'Воитель'}, 
                   {'text': 'Достичь максимального уровня в игре', 'title': '...'}, 
                   {'text': 'Пройти основной сюжет игры', 'title': 'Воитель'}, 
                   {'text': 'Убить 100 противников', 'title': 'Убийца'}, '...'])

+  where          [{'text': 'Пройти основной сюжет игры', 'title': 'Воитель'},
                   {'text': 'Достичь максимального уровня в игре', 'title': '...'}, 
                   {'text': 'Пройти основной сюжет игры', 'title': 'Воитель'},
                   {'text': 'Убить 100 противников', 'title': 'Убийца'}, '...'] = <solution.FullNotificationPrinter object at 0x7f854bf366a0>.achievements

+  and    5 = len([{'text': 'Пройти основной сюжет игры', 'title': 'Воитель'}, 
                   {'text': 'Достичь максимального уровня в игре', 'title': '...ца'},
                   {'text': 'Выучить все заклинания', 'title': 'Мерлин'}, 
                   {'text': 'Победить 1000 противников', 'title': 'Чемпион'}])


AssertionError: Тест 12.3. Не верное сохранение достижений у подписчика FullNotificationPrinter после получения уведомлений. 
                Каждое достижение должно быть уникальным (то есть учтено только один раз).

assert 128 == 8
+  where 128 = len([{'text': 'Дается за выполнение основного квеста в игре', 'title': 'Покоритель'}, 
                 {'text': 'Пройти основной сюжет игры...о'}, 
                 {'text': 'Выучить все заклинания', 'title': 'Мерлин'}, 
                 {'text': 'Выучить все заклинания', 'title': 'Мерлин'}, ...])

+  where [{'text': 'Дается за выполнение основного квеста в игре', 'title': 'Покоритель'}, 
       {'text': 'Пройти основной сюжет игры...о'}, 
       {'text': 'Выучить все заклинания', 'title': 'Мерлин'}, 
       {'text': 'Выучить все заклинания', 'title': 'Мерлин'}, ...] = <solution.FullNotificationPrinter object at 0x7f9d50a52e48>.achievements

+  and   8 = len([{'text': 'Дается за выполнение основного квеста в игре', 'title': 'Покоритель'}, 
               {'text': 'Пройти основной сюжет игры... 'Найти все предметы в игре', 'title': 'Коллекционер'}, 
               {'text': 'Победить 1000 противников', 'title': 'Чемпион'}, ...])

+  where [{'text': 'Дается за выполнение основного квеста в игре', 'title': 'Покоритель'}, 
       {'text': 'Пройти основной сюжет игры... 'Найти все предметы в игре', 'title': 'Коллекционер'}, 
       {'text': 'Победить 1000 противников', 'title': 'Чемпион'}, ...] = Engine.achievements_list
'''
