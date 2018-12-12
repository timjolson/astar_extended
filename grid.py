#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import cv2
from copy import copy
import itertools
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import h3bot
from h3bot import Position
from h3bot.entity import Ship
from h3bot.positionals import get_position

from h3bot.astar_extended import astar_multiagent, astar_multiagent_no_cache, astar, dijkstras, greedy


# In[19]:


np.random.seed(12345+12345)
size = 32
Position.width = Position.height = size

MOVE_COST_RATIO = 10
EXTRACT_RATIO = 4
MAX_HALITE = 1000

map = np.random.random_sample((size, size)) * MAX_HALITE


# In[20]:


def neighbors(pos):
    print(f"neighbors {pos}")
    for dx, dy in itertools.product([-1,0,1],[-1,0,1]):
        if not (dx and dy):
#             print(f"neighbors{pos}")
#             print(f"yield {pos.step+1}")
#             print(f"yield {pos+(dx,dy)}")
            yield Node(pos.step+1, (pos+(dx,dy)).norm())

def dist(pos, next):
    print(f"dist{pos, next}")
    direct = get_position(pos) - next
    move = (min(direct.x, size-direct.x), min(direct.y, size-direct.y))
    return (move[0] + move[1])

def cost(pos, next):
    print(f"cost{pos, next}")
    nvisits[0] += 1
    visited[next] += 0.2
    return (map[pos] / MOVE_COST_RATIO) * int(bool(pos!=next)) + dist(next, home) * HOME_COST

def heur(pos, next):
    print(f"heur{pos, next}")
    nvisits[0] += 1
    visited[next] += 0.2
    if pos != next:
        return (map[next] * EXTRACT_RATIO) - dist(next, home) * HOME_COST * MAX_HALITE
    else:
        return (map[next] * EXTRACT_RATIO) - dist(pos, home) * HOME_COST * MAX_HALITE

def draw_path(img, path):
    if path[0] is not None:
        for node in path:
            img[node] = 1.0
            visited[node] = 0

class Node(Position):
    def __new__(cls, step, x, y=None):
        if y is not None:
            x = (x,y)
        self = super().__new__(cls, x)
        return self
    
    def __init__(self, step, *args):
#         print(f"args {args}")
#         self.position = get_position(pos)
        super().__init__(*args)
        self.step = step
        print(self)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.step) + ', ' + super().__repr__()[1:-1] + ')'

    def __eq__(self, other):
        return self.step == other.step and super().__eq__(self)
#         if isinstance(other, entity.Entity):
#             other = other.position
        return self.x == other[0] and self.y == other[1]
    
    def __hash__(self):
        return hash((self.x, self.y, self.step))

class SimShip(Ship):
    def __init__(self, pos, halite, step):
        super().__init__(0, 0, pos, halite)
        print(f"SimShip{pos, halite, step}")
        self.step = step

Node(0,1,2)==Node(0,1,2), Node(0, (1,2)) == Node(0, (1,2)), Node(0,1,2)==Node(1,1,2), Node(0, (1,2)) == Node(1, (1,2))


# In[21]:


# ship = Position(np.random.randint(0, size), np.random.randint(0, size))
ship = SimShip(Node(0, (20,18)), 0, 0)
# home = (np.random.randint(0, size), np.random.randint(0, size))
home = Node(0,(1,2))

nvisits = [0]
HOME_COST = 1


# In[26]:


blank = np.zeros((size,size))
checks = np.zeros_like(map)
paths = np.zeros_like(map)
visited = np.zeros_like(map)
targets = np.zeros_like(map)
nvisits[0] = 0

start, end = home, ship.position
print(f"neighbors {list(neighbors(ship.position))}")

path = astar(start, goal=end,
    neighbors_fnct=neighbors,
    distance_between_fnct=cost,
    heuristic_cost_estimate_fnct=heur
)
draw_path(paths, path[start])
print(f"{len(path[start])} steps")
print(f"{len(set(path[start]))} unique nodes")
print(f"% visited {100*nvisits[0]/(size*size)}")

visited[start] = 1.0
visited[end] = 1.0
targets[start] = 1.0
targets[end] = 1.0

halite = 0
for n in path[start][1:]:
    halite += map[n] / EXTRACT_RATIO
    halite -= map[n] / MOVE_COST_RATIO
print(f"{halite} halite")

print(path[start])

plt.subplots(1,2, figsize=(15,15))

plt.subplot(1,2,1)
# rgb
image = np.dstack((visited, blank, paths))
plt.imshow(image)

plt.subplot(1,2,2)
# rgb
image = np.dstack((blank, map/MAX_HALITE, targets))
plt.imshow(image)

plt.show()


# In[ ]:




