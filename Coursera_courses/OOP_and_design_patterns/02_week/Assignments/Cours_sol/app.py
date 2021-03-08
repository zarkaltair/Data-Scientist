import pygame
import random
from polylines import Polyline, Knot
from vector import Vec2d

SCREEN_DIM = (800, 600)


def draw_help(lines, current):
    gameDisplay.fill((50, 50, 50))
    font1 = pygame.font.SysFont("courier", 24)
    font2 = pygame.font.SysFont("serif", 24)
    data = list()
    data.append(["F1", "Show Help"])
    data.append(["R", "Restart"])
    data.append(["P", "Pause/Play"])
    data.append(["Num+", "More points"])
    data.append(["Num-", "Less points"])
    data.append(["1 - 0", "Change line (current not white)"])
    data.append(["Del", "Remove last point (current not white)"])
    data.append(["Num*", "Increase speed"])
    data.append(["Num/", "Decrease speed"])
    data.append(["————————————————————————————————————", ""])
    data.append(["", "Current knot is " + str(current)])
    data.append([str(lines[current].addition_points), "addition points of current knot"])

    pygame.draw.lines(gameDisplay, (255, 50, 50, 255), True, [
                      (0, 0), (800, 0), (800, 600), (0, 600)], 5)
    for i, text in enumerate(data):
        gameDisplay.blit(font1.render(
            text[0], True, (128, 128, 255)), (100, 100 + 30 * i))
        gameDisplay.blit(font2.render(
            text[1], True, (128, 128, 255)), (200, 100 + 30 * i))


if __name__ == "__main__":
    pygame.init()
    gameDisplay = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("MyScreenSaver")

    working = True
    knots = [Polyline()] + [Knot() for i in range(9)]
    current_knot = 1
    show_help = False
    pause = True
    speed = 1

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
                    knots = [Knot() for i in range(9)]
                if event.key == pygame.K_p:
                    pause = not pause
                if event.key == pygame.K_KP_PLUS:
                    knots[current_knot].addition_points += 1
                if event.key == pygame.K_F1:
                    show_help = not show_help
                if event.key == pygame.K_KP_MULTIPLY:
                    speed *= 2
                if event.key == pygame.K_KP_DIVIDE:
                    speed /= 2
                if event.key == pygame.K_KP_MINUS:
                    knots[current_knot].addition_points -= 1
                if event.key in range(48, 59):  # event.key is number
                    knots[current_knot].color = pygame.Color("white")
                    current_knot = event.key - 48
                if event.key == pygame.K_DELETE:
                    knots[current_knot].remove_point()

            if event.type == pygame.MOUSEBUTTONDOWN:
                knots[current_knot].add_point(Vec2d(*event.pos), Vec2d(
                    random.random() * 2, random.random() * 2))

        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        knots[current_knot].color.hsla = (hue, 100, 50, 100)
        for k in knots:
            k.draw_points(gameDisplay, 'line')

            if not pause:
                k.set_points(*SCREEN_DIM, speed)
        if show_help:
            draw_help(knots, current_knot)

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
