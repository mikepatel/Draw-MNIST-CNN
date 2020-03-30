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


################################################################################
# Main
if __name__ == "__main__":
    # load classifier model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved_model"))

    pygame.init()

    # set up drawing window
    screen = pygame.display.set_mode(size=[SCREEN_WIDTH, SCREEN_HEIGHT])

    while not DONE:
        for event in pygame.event.get():
            # close window button
            if event.type == pygame.QUIT:
                DONE = True

            # clear screen
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT_CLICK:
                screen.fill(BLACK)

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

                # save current screen image to feed into classifier
                saved_image_filepath = os.path.join(os.getcwd(), "saved_images\\saved.png")
                pygame.image.save(screen, saved_image_filepath)
                image = Image.open(saved_image_filepath).convert("L")  # convert to greyscale

                # resize image
                image = image.resize((28, 28))

                # normalize image
                image = np.array(image).astype(np.float32)

                # reshape: (1, 28, 28, 1)
                image = image.reshape(-1, 28, 28, 1)

                # classify saved image
                prediction = model.predict(image)
                print(np.argmax(prediction))

                # display prediction in secondary screen

            pygame.display.flip()

    # done
    pygame.quit()
