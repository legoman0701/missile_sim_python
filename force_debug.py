from missle_siml_2d import *
import sys, random
from math import *

import pygame
from pygame.locals import *
import numpy as np
from numpy import array as vec

SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600
SCREEN_SIZE   = vec([SCREEN_WIDTH, SCREEN_HEIGHT])
FPS           = 60

BLACK = (0, 0, 0)

canard = Wing(0.012, 4.0, 0.01, 2, 0.7)
fin = Wing(0.025, 4.5, 0.015, 2.5, 0.8)

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
debug_surface = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Missile Simulation 2D")

debug_font = pygame.font.SysFont('Arial', 11, bold=True)

clock  = pygame.time.Clock()
cam    = Camera()
missle = MissileSimulation()
target = Target([-3000.0, -3000.0])
running = True

t = 0.0

while running:
    dt = clock.tick(FPS) / 1000.0
    dt = 1/FPS
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    missle.pos += vec([cos(0), sin(0)])
    print(missle.pos)

    screen.fill((0, 0, 0))
    pygame.dra

    pygame.display.flip()
