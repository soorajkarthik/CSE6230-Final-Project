#script to visualize clustering output
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <file>")
    exit()

path = sys.argv[1]
file = open(path, "r")

points, centroids, assignments = [], [], []

n_points, n_dims = [int(s) for s in re.split("\\s+", file.readline().strip())]

if n_dims != 2:
    print(f"Can only visualize 2d data, but got {n_dims} dimensions")
    exit()

for _ in range(n_points):
    points.append([float(s) for s in re.split("\\s+", file.readline().strip())])

n_centroids, n_dims = [int(s) for s in re.split("\\s+", file.readline().strip())]

for _ in range(n_centroids):
    centroids.append([float(s) for s in re.split("\\s+", file.readline().strip())])

assignments = [int(s) for s in re.split("\\s+", file.readline().strip())]

points = np.array(points)
centroids = np.array(centroids)

colors = cm.get_cmap('hsv', n_centroids)

plt.scatter(points[:, 0], points[:, 1], c=assignments, cmap=colors)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black')

plt.show()