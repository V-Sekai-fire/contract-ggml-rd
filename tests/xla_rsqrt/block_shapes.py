import numpy as np
from floor_of_line import tables
T = tables()
for e in (127, 128):
    y = T[e]
    for G in (4, 8, 16, 32, 64):
        n = 4096 // G
        pats = {}
        for j in range(n):
            p = tuple((y[j * G:(j + 1) * G] - y[j * G]).tolist())
            pats[p] = pats.get(p, 0) + 1
        print("e=%d G=%2d: %d blocks, %d distinct shapes, most common covers %d" % (e, G, n, len(pats), max(pats.values())))
