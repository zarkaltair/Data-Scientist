#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import random

SCREEN_DIM = (800, 600)


# =======================================================================================
# Функции для работы с векторами
# =======================================================================================
class Vec2d:
    def __init__(self, v):
        self.v = v

    def __sub__(self, obj):
        """"возвращает разность двух векторов"""
        return self.v[0] - obj[0], self.v[1] - obj[1]


    def __add__(self, obj):
        """возвращает сумму двух векторов"""
        return self.v[0] + obj[0], self.v[1] + obj[1]


    def __mul__(self, obj):
        """возвращает произведение вектора на число"""
        return self.v[0] * obj, self.v[1] * obj


    # def length(self, x):
    #     """возвращает длину вектора"""
    #     return (self.x[0] * self.x[0] + self.x[1] * self.x[1]) ** 0.5


    # def vec(self, x, y):
    #     """возвращает пару координат, определяющих вектор (координаты точки конца вектора),
    #     координаты начальной точки вектора совпадают с началом системы координат (0, 0)"""
    #     return sub(self.y, self.x)

    # def int_pair(self, x, y):
    #     """возвращает кортеж из двух целых чисел (текущие координаты вектора)"""
    #     pass


# =======================================================================================
# Функции отрисовки
# =======================================================================================
class Polyline:
    def draw_points(self, points, style='points', width=3, color=(255, 255, 255)):
        """функция отрисовки точек на экране"""
        if style == "line":
            for p_n in range(-1, len(points) - 1):
                pygame.draw.line(gameDisplay, color,
                                 (int(points[p_n][0]), int(points[p_n][1])),
                                 (int(points[p_n + 1][0]), int(points[p_n + 1][1])), width)

        elif style == "points":
            for p in points:
                pygame.draw.circle(gameDisplay, color, (int(p[0]), int(p[1])), width)


    def draw_help(self):
        """функция отрисовки экрана справки программы"""
        gameDisplay.fill((50, 50, 50))
        font1 = pygame.font.SysFont("courier", 24)
        font2 = pygame.font.SysFont("serif", 24)
        data = []
        data.append(["F1", "Show Help"])
        data.append(["R", "Restart"])
        data.append(["P", "Pause/Play"])
        # data.append(["Num+", "More points"])
        # data.append(["Num-", "Less points"])
        data.append(["", ""])
        data.append([str(steps), "Current points"])

        pygame.draw.lines(gameDisplay, (255, 50, 50, 255), True, [
            (0, 0), (800, 0), (800, 600), (0, 600)], 5)
        for i, text in enumerate(data):
            gameDisplay.blit(font1.render(
                text[0], True, (128, 128, 255)), (100, 100 + 30 * i))
            gameDisplay.blit(font2.render(
                text[1], True, (128, 128, 255)), (200, 100 + 30 * i))


    # =======================================================================================
    # Функции, отвечающие за расчет сглаживания ломаной
    # =======================================================================================
    def get_point(self, points, alpha, deg=None):
        # print('_____________iter rec_____________')
        if deg is None:
            deg = len(points) - 1
        if deg == 0:
            return points[0]
        # print('deg = ', deg)
        v1 = Vec2d(points[deg]) * alpha
        # print('v1 =', v1)
        # print(points)
        v2 = self.get_point(points, alpha, deg - 1)
        # print('v2 =', v2)
        return Vec2d(v1) + Vec2d(v2) * (1 - alpha)


    def get_points(self, base_points, count):
        alpha = 1 / count
        res = []
        for i in range(count):
            res.append(self.get_point(base_points, i * alpha))
        return res


    def set_points(self, points, speeds):
        """функция перерасчета координат опорных точек"""
        for p in range(len(points)):
            points[p] = Vec2d(points[p]) + speeds[p]
            if points[p][0] > SCREEN_DIM[0] or points[p][0] < 0:
                speeds[p] = (- speeds[p][0], speeds[p][1])
            if points[p][1] > SCREEN_DIM[1] or points[p][1] < 0:
                speeds[p] = (speeds[p][0], -speeds[p][1])


class Knot(Polyline):
    def get_knot(self, points, count):
        if len(points) < 3:
            return []
        res = []
        for i in range(-2, len(points) - 2):
            ptn = []
            ptn.append((Vec2d(Vec2d(points[i]) + points[i + 1]) * 0.5))
            ptn.append(Vec2d(points[i + 1]) * 1)
            ptn.append((Vec2d(Vec2d(points[i + 1]) + points[i + 2]) * 0.5))

            res.extend(self.get_points(ptn, count))
        return res


# =======================================================================================
# Основная программа
# =======================================================================================
if __name__ == "__main__":
    pygame.init()
    gameDisplay = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("MyScreenSaver")

    steps = 10
    working = True
    points = []
    speeds = []
    show_help = False
    pause = True
    style = 'points'
    width = 3
    hue = 0
    color = pygame.Color(0)
    a = Polyline()
    b = Knot()

    while working:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                working = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    working = False
                if event.key == pygame.K_r:
                    points = []
                    speeds = []
                if event.key == pygame.K_p:
                    pause = not pause
                if event.key == pygame.K_KP_PLUS:
                    steps += 1
                if event.key == pygame.K_F1:
                    show_help = not show_help
                if event.key == pygame.K_KP_MINUS:
                    steps -= 1 if steps > 1 else 0

            if event.type == pygame.MOUSEBUTTONDOWN:
                points.append(event.pos)
                speeds.append((random.random() * 2, random.random() * 2))

        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        color.hsla = (hue, 100, 50, 100)
        a.draw_points(points)
        a.draw_points(b.get_knot(points, steps), "line", 3, color)
        if not pause:
            a.set_points(points, speeds)
        if show_help:
            a.draw_help()

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
