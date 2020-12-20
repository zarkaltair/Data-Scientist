import re
from abc import ABC, abstractmethod


# create system
class System:
    def __init__(self, text):
        tmp = re.sub(r'\W', ' ', text.lower())
        tmp = re.sub(r' +', ' ', tmp).strip()
        self.text = tmp

    def get_processed_text(self, processor):
        result = processor.process_text(self.text)
        print(*result, sep='\n')


# abstract class with abstract method
class TextProcessor(ABC):
    @abstractmethod
    def process_text(self, text):
        pass


# counter words
class WordCounter:
    def count_words(self, text):
        self.__words = dict()
        for word in text.split():
            self.__words[word] = self.__words.get(word, 0) + 1

    def get_count(self, word):
        return self.__words.get(word, 0)

    def get_all_words(self):
        return self.__words.copy()


class WordCounterAdapter(TextProcessor):
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def process_text(self, text):
        self.adaptee.count_words(text)
        words = self.adaptee.get_all_words().keys()
        return sorted(words,
                      key=lambda x: self.adaptee.get_count(x),
                      reverse=True)


text = 'В прошлом видео мы разобрались с паттерном проектирования Adapter. Поняли, зачем он применятся и в каких задачах используется. Давайте сейчас на практике разберемся, как реализовать паттерн Adapter с использованием языка программирования Python. Пусть у нас есть некоторая система, которая берет какой-то текст, делает его предварительную обработку, а дальше хочет вывести слова в порядке убывания их частоты. Но собственного обработчика у системы нету. Она принимает в качестве обработчика некоторый объект, который имеет интерфейс доступа, который вы видите сейчас на экране.'

# create instance of System
system = System(text)
# print(system)

# create instance of WordCounter
counter = WordCounter()
# raise AttributeError: 'WordCounter' object has no attribute 'process_text'
# system.get_processed_text(counter)

# create adapter for system
adapter = WordCounterAdapter(counter)
system.get_processed_text(adapter)
