"""
Michael Patel
March 2020

Project description:
    Build an interactive MNIST classifier using pygame

File description:
    For pygame app and inference

"""
################################################################################
# Imports
import os
import pygame


################################################################################
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RADIUS = 7
DRAW = False
LAST_POS = (0, 0)
DONE = False


################################################################################
# roundline
def roundline(surface, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(surface, color, (x, y), radius)


################################################################################
# Main
if __name__ == "__main__":
    pygame.init()

    # set up drawing window
    screen = pygame.display.set_mode(size=[SCREEN_WIDTH, SCREEN_HEIGHT])

    while not DONE:
        for event in pygame.event.get():
            # close window button
            if event.type == pygame.QUIT:
                DONE = True

            # start drawing using mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.circle(screen, WHITE, event.pos, RADIUS)
                DRAW = True

            # continue drawing
            if event.type == pygame.MOUSEMOTION:
                if DRAW:
                    pygame.draw.circle(screen, WHITE, event.pos, RADIUS)
                    roundline(screen, WHITE, event.pos, LAST_POS, RADIUS)
                LAST_POS = event.pos

            # stop drawing
            if event.type == pygame.MOUSEBUTTONUP:
                DRAW = False

                # save current screen image to feed into classifier
                pygame.image.save(screen, os.path.join(os.getcwd(), "saved_images\\input.png"))

                # classify saved image

            pygame.display.flip()

    # done
    pygame.quit()
