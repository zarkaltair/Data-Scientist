import pygame


class Polyline:

    def __init__(self, color=None):
        self.__points = []
        self.__speeds = []
        self.__N = 0
        self.color = pygame.Color(*(color or (255, 255, 255)))

    def add_point(self, point, speed):
        self.__points.append(point)
        self.__speeds.append(speed)
        self.__N += 1

    def add_points(self, points, speeds):
        self.__points.extend(points)
        self.__speeds.extend(speeds)
        self.__N += len(points)

    def remove_point(self):
        self.__points.pop()
        self.__speeds.pop()
        self.__N -= 1

    def clear(self):
        self.__points = []
        self.__speeds = []
        self.__N = 0

    def set_points(self, screen_width, screen_height, mul):
        for p in range(self.__N):
            self.__points[p] += mul * self.__speeds[p]
            if self.__points[p].x > screen_width or self.__points[p].x < 0:
                self.__speeds[p].x = - self.__speeds[p].x
            if self.__points[p].y > screen_height or self.__points[p].y < 0:
                self.__speeds[p].y = - self.__speeds[p].y

    # "Отрисовка" точек
    def draw_points(self, display, style="points", width=3):
        if style == "line":
            for p_n in range(-1, self.__N - 1):
                pygame.draw.line(display, self.color, self.__points[p_n].int_pair(
                ), self.__points[p_n + 1].int_pair(), width)

        elif style == "points":
            for p in self.__points:
                pygame.draw.circle(display, self.color, p.int_pair(), width)


# Сглаживание ломаной

class Knot(Polyline):

    def __init__(self, color=None, addition_points=5):
        Polyline.__init__(self)
        self.__points = []
        self.__speeds = []
        self.__N = 0
        self.__count = addition_points
        self.color = pygame.Color(*(color or (255, 255, 255)))

    def __set_count(self, x):
        self.__count = x if 0 < x < 30 else self.__count
        self.__get_knot()

    def __get_count(self):
        return self.__count

    addition_points = property(__get_count, __set_count)

    def __get_point(self, points, alpha, deg=None):
        if deg is None:
            deg = len(points) - 1
        if deg == 0:
            return points[0]
        return points[deg] * alpha + self.__get_point(points, alpha, deg - 1) * (1 - alpha)

    def __get_points(self, base_points):
        alpha = 1 / self.__count
        res = []
        for i in range(self.__count):
            res.append(self.__get_point(base_points, i * alpha))
        return res

    def __get_knot(self):
        Polyline.clear(self)
        if len(self.__points) < 3:
            return []
        for i in range(-2, len(self.__points) - 2):
            ptn = list()
            ptn.append((self.__points[i] + self.__points[i+1]) * 0.5)
            ptn.append(self.__points[i + 1])
            ptn.append((self.__points[i+1] + self.__points[i+2]) * 0.5)

            Polyline.add_points(self, self.__get_points(ptn), [])

    def add_point(self, point, speed):
        self.__points.append(point)
        self.__speeds.append(speed)
        self.__N += 1
        self.__get_knot()

    def remove_point(self):
        self.__points.pop()
        self.__speeds.pop()
        self.__N -= 1
        self.__get_knot()

    def set_points(self, screen_width, screen_height, mul):
        for p in range(self.__N):
            self.__points[p] += mul * self.__speeds[p]
            if self.__points[p].x > screen_width or self.__points[p].x < 0:
                self.__speeds[p].x = - self.__speeds[p].x
            if self.__points[p].y > screen_height or self.__points[p].y < 0:
                self.__speeds[p].y = - self.__speeds[p].y
        self.__get_knot()
