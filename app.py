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
import numpy as np
import pygame
from PIL import Image

import tensorflow as tf


################################################################################
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
GREEN = [0, 128, 0]
RADIUS = 25
DRAW = False
LAST_POS = (0, 0)
DONE = False
LEFT_CLICK = 1
RIGHT_CLICK = 3


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


# divide screen in half (draw, prediction)
def divide_screen(surface):
    pygame.draw.line(surface, GREEN, [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], 10)


################################################################################
# Main
if __name__ == "__main__":
    # load classifier model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved_model"))

    pygame.init()

    # set up drawing window
    screen = pygame.display.set_mode(size=[2*SCREEN_WIDTH, SCREEN_HEIGHT])
    divide_screen(screen)

    while not DONE:
        for event in pygame.event.get():
            # close window button
            if event.type == pygame.QUIT:
                DONE = True

            # clear screen
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT_CLICK:
                screen.fill(BLACK)
                divide_screen(screen)

            # start drawing using mouse
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT_CLICK:
                pygame.draw.circle(screen, WHITE, event.pos, RADIUS)
                DRAW = True

            # continue drawing
            if event.type == pygame.MOUSEMOTION:
                if DRAW:
                    pygame.draw.circle(screen, WHITE, event.pos, RADIUS)
                    roundline(screen, WHITE, event.pos, LAST_POS, RADIUS)
                LAST_POS = event.pos

            # stop drawing
            if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT_CLICK:
                DRAW = False

                # crop and save drawn portion
                saved_image_filepath = os.path.join(os.getcwd(), "saved_images\\saved.png")
                drawn_image = pygame.Surface((SCREEN_WIDTH-10, SCREEN_HEIGHT-10))
                drawn_image.blit(screen, (0, 0), (0, 0, SCREEN_WIDTH-10, SCREEN_HEIGHT-10))
                pygame.image.save(drawn_image, saved_image_filepath)

                # feed saved image into classifier
                image = Image.open(saved_image_filepath).convert("L")  # convert to greyscale

                # resize image
                image = image.resize((28, 28))

                # turn image into an array
                image = np.array(image).astype(np.float32)

                # reshape: (1, 28, 28, 1)
                image = image.reshape(-1, 28, 28, 1)

                # classify saved image
                prediction = model.predict(image)
                label = np.argmax(prediction)

                # display prediction in right half of the screen
                font = pygame.font.SysFont("Arial", 144)
                text_surface = font.render(str(label), False, GREEN)
                screen.blit(text_surface, (int(SCREEN_WIDTH*1.5), int(SCREEN_HEIGHT*0.35)))

            pygame.display.flip()

    # done
    pygame.quit()
