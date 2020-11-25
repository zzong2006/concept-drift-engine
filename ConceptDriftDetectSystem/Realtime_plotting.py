import numpy as np
import matplotlib.pyplot as plt


def uniqueish_color():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())


xy = (np.random.random((10, 2)) - 0.5).cumsum(axis=0)

fig, ax = plt.subplots()
for start, stop in zip(xy[:-1], xy[1:]):
    x, y = zip(start, stop)
    ax.plot(x, y, color=uniqueish_color())
plt.show()

'''
If you're plotting something with a million line segments, though, 
this will be terribly slow to draw. 
In that case, use a LineCollection. E.g.
'''

from matplotlib.collections import LineCollection

xy = (np.random.random((1000, 2)) - 0.5).cumsum(axis=0)

# Reshape things so that we have a sequence of:
# [[(x0,y0),(x1,y1)],[(x0,y0),(x1,y1)],...]
xy = xy.reshape(-1, 1, 2)
segments = np.hstack([xy[:-1], xy[1:]])

fig, ax = plt.subplots()
coll = LineCollection(segments, cmap=plt.cm.gist_ncar)
coll.set_array(np.random.random(xy.shape[0]))

ax.add_collection(coll)
ax.autoscale_view()

plt.show()

# !/usr/bin/env python
'''
Color parts of a line based on its properties, e.g., slope.
'''
from matplotlib.colors import ListedColormap, BoundaryNorm

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)
z = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# Create a colormap for red, green and blue and a norm to color
# f' < -0.5 red, f' > 0.5 blue, and the rest green
cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the line collection object, setting the colormapping parameters.
# Have to set the actual values used for colormapping separately.
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(z)
lc.set_linewidth(3)
plt.gca().add_collection(lc)

plt.xlim(x.min(), x.max())
plt.ylim(-1.1, 1.1)
plt.show()
