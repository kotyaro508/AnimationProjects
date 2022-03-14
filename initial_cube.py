from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1] - r[0]:
        ax.plot3D(*zip(s, e), color="grey")

# draw a point
ax.scatter([0], [0], [0], color="k", s=100)

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D([0, 1], [0, 0], [0, 0], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="g")
ax.add_artist(a)
b = Arrow3D([0, 0], [0, 1], [0, 0], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="b")
ax.add_artist(b)
c = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="r")
ax.add_artist(c)
plt.show()
