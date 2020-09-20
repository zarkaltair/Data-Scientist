from abc import ABCMeta, abstractmethod


class Sender(metaclass=ABCMeta):
    @abstractmethod
    def send(self):
        '''Do something'''


class Child(Sender):
    def send(self):
        print('Sending')

Child()


class PythonWay:

    def send(self):
        raise NotImplementedError
