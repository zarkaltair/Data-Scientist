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

    def __len__(self):
        """возвращает длину вектора"""
        return (self.x[0] ** 2 + self.x[1] ** 2) ** 0.5

    def vec(self, obj):
        """возвращает пару координат, определяющих вектор (координаты точки конца вектора),
        координаты начальной точки вектора совпадают с началом системы координат (0, 0)"""
        return sub(self.v[0], self.v[1])

    def int_pair(self):
        """возвращает кортеж из двух целых чисел (текущие координаты вектора)"""
        return self.v[0], self.v[1]


# =======================================================================================
# Функции отрисовки
# =======================================================================================
class Polyline:
    def __init__(self, points, speeds):
        self.points = points
        self.speeds = speeds

    def draw_points(self, style='points', width=3, color=(255, 255, 255)):
        """функция отрисовки точек на экране"""
        if style == "line":
            for p_n in range(-1, len(self.points) - 1):
                pygame.draw.line(gameDisplay, color,
                                 (int(self.points[p_n][0]), int(self.points[p_n][1])),
                                 (int(self.points[p_n + 1][0]), 
                                  int(self.points[p_n + 1][1])), width)

        elif style == "points":
            for p in self.points:
                pygame.draw.circle(gameDisplay, color, (int(p[0]), int(p[1])), width)

    # =======================================================================================
    # Функции, отвечающие за расчет сглаживания ломаной
    # =======================================================================================
    def get_point(self, base_points, alpha, deg=None):
        if deg is None:
            deg = len(base_points) - 1
        if deg == 0:
            return base_points[0]
        v1 = Vec2d(base_points[int(deg)]) * alpha
        v2 = self.get_point(base_points, alpha, deg - 1)
        return Vec2d(v1) + Vec2d(v2) * (1 - alpha)

    def get_points(self, base_points, count):
        alpha = 1 / count
        res = []
        for i in range(count):
            res.append(self.get_point(base_points, i * alpha))
        return res

    def set_points(self):
        """функция перерасчета координат опорных точек"""
        for p in range(len(self.points)):
            self.points[p] = Vec2d(self.points[p]) + self.speeds[p]
            if self.points[p][0] > SCREEN_DIM[0] or self.points[p][0] < 0:
                self.speeds[p] = (-self.speeds[p][0], self.speeds[p][1])
            if self.points[p][1] > SCREEN_DIM[1] or self.points[p][1] < 0:
                self.speeds[p] = (self.speeds[p][0], -self.speeds[p][1])


class Knot(Polyline):
    def __init__(self, points, count=0):
        self.points = points
        self.count = count

    def get_knot(self):
        if len(self.points) < 3:
            return []
        res = []
        for i in range(-2, len(self.points) - 2):
            ptn = []
            ptn.append((Vec2d(Vec2d(self.points[i]) + self.points[i + 1]) * 0.5))
            ptn.append(Vec2d(self.points[i + 1]) * 1)
            ptn.append((Vec2d(Vec2d(self.points[i + 1]) + self.points[i + 2]) * 0.5))

            res.extend(self.get_points(ptn, self.count))
        return res


def draw_help():
    """функция отрисовки экрана справки программы"""
    font1 = pygame.font.SysFont("courier", 24)
    font2 = pygame.font.SysFont("serif", 24)

    data = []
    data.append(["F1", "Show Help"])
    data.append(["R", "Restart screen saver"])
    data.append(["P", "Pause/Play"])
    data.append(["M", "Add transitional point"])
    data.append(["N", "Remove transitional points"])
    data.append(["U", "Speed Up"])
    data.append(["D", "Speed Down"])
    data.append(["T", "Add anchor point"])
    data.append(["Y", "Remove anchor point"])
    data.append(["", ""])
    data.append([str(steps), "Current points"])

    surf = pygame.Surface((800, 600))
    surf.fill((50, 50, 50))
    surf.set_alpha(200)
    
    pygame.draw.lines(surf, (0, 250, 0, 255), True, [
        (0, 0), (800, 0), (800, 600), (0, 600)], 3)

    for i, text in enumerate(data):
        surf.blit(font1.render(
            text[0], True, (128, 128, 255)), (100, 100 + 30 * i))
        surf.blit(font2.render(
            text[1], True, (128, 128, 255)), (200, 100 + 30 * i))

    gameDisplay.blit(surf, (0, 0))


# =======================================================================================
# Основная программа
# =======================================================================================
if __name__ == "__main__":
    pygame.init()
    gameDisplay = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("My Screen Saver")

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
                if event.key == pygame.K_m:
                    steps += 1
                if event.key == pygame.K_F1:
                    show_help = not show_help
                if event.key == pygame.K_n:
                    steps -= 1 if steps > 1 else 0
                if event.key == pygame.K_u:
                    speeds = [tuple(j * 2 for j in i) for i in speeds]
                if event.key == pygame.K_d:
                    speeds = [tuple(j * 0.5 for j in i) for i in speeds]
                if event.key == pygame.K_t:
                    points.append((400, 300))
                    speeds.append((random.random() * 2, random.random() * 2))
                if event.key == pygame.K_y:
                    try:
                        points.pop()
                        speeds.pop()
                    except:
                        pass

            if event.type == pygame.MOUSEBUTTONDOWN:
                points.append(event.pos)
                speeds.append((random.random() * 2, random.random() * 2))


        polyline = Polyline(points, speeds)
        anchor = Knot(polyline.points, steps)
        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        color.hsla = (hue, 100, 50, 100)
        polyline.draw_points()
        spline = Polyline(anchor.get_knot(), speeds)
        spline.draw_points(style="line", color=color)
        if not pause:
            polyline.set_points()
        if show_help:
            draw_help()

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
