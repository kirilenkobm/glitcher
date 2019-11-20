#!/usr/bin/env python3
import sys

try:
    N = int(sys.argv[1])
except (IndexError, ValueError):
    sys.exit(f"Usage: {sys.argv[0]} [N]")

matrix = [[0 for _ in range(N)] for _ in range(N)]

pos = [0, 0]
num = 1
direction = "RIGHT"
r_border = N
d_border = N
l_border = 0
u_border = 1

for step in range(N ** 2):
    matrix[pos[0]][pos[1]] = num
    if direction == "RIGHT":
        pos[1] += 1
        if pos[1] == r_border - 1:
            direction = "DOWN"
            r_border -= 1
    elif direction == "DOWN":
        pos[0] += 1
        if pos[0] == d_border - 1:
            direction = "LEFT"
            d_border -= 1
    elif direction == "LEFT":
        pos[1] -= 1
        if pos[1] == l_border:
            direction = "UP"
            l_border += 1
    elif direction == "UP":
        pos[0] -= 1
        if pos[0] == u_border:
            direction = "RIGHT"
            u_border += 1
    num += 1

for elem in matrix:
    print(" ".join(map(str, elem)))
