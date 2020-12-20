from abc import ABC, abstractmethod


class ChessPiece(ABC):
    # общий метод, который будут использовать все наследники этого класса
    def draw(self):
        print("Drew a chess piece")

    # абстрактный метод, который будет необходимо переопределять для каждого подкласса
    @abstractmethod
    def move(self):
        pass


class Queen(ChessPiece):
    def move(self):
        print("Moved Queen to e2e4")


# Мы можем создать экземпляр класса
q = Queen()
# И нам доступны все методы класса
q.draw()
q.move()
