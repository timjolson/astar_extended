from astar_extended import AStar, MultiagentAStar, CachedAStar
import numpy as np
import random
import cv2
import sys
import logging
import copy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')
# logging.basicConfig(filename='astar.log', filemode='w', level=logging.INFO, format='%(message)s')
np.random.seed(123)

MOVE_COST_RATIO = 10
EXTRACT_RATIO = 4
MAX_HALITE = 1000

def onmouse(window, data, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        logging.info(f"Clicked on {window}: {x,y} {data[y,x]}")
        cv2.setWindowTitle(window, f"Clicked on {window}: {x,y} {data[y,x]}")


# size = np.random.randint(32, 64)
size = 32
import hlt
hlt.networking.Game.width, hlt.networking.Game.height = size, size
Position = hlt.Position
Ship = hlt.entity.Ship
get_position = hlt.positionals.get_position

from sim import Node

map = (np.random.random_sample((size, size)) * MAX_HALITE)
paths = np.zeros_like(map)
checks = np.zeros_like(map)
visited = np.zeros_like(map)
points = np.zeros_like(map)
paths = np.zeros_like(map)
map = map.astype(np.int)


# ship = Ship(0, 0, Node(np.random.randint(0, size), np.random.randint(0, size)), 0)
ship = Ship(0, 0, Node(19, 4), 0)
# home = Position(np.random.randint(0, size), np.random.randint(0, size))
# home = Position(5, 22)
home = Node(5, 22)

map[ship.position] = 0
map[home] = 0

TURN_LIMIT = int(size*2.0)

logging.info(f"{ship}\nHome {home}")
logging.info(f"Distance {hlt.networking.Game.calculate_distance(ship, home)} :: Turns {TURN_LIMIT}")


def neighbors(map, goal, pos):
    if pos.turn < (TURN_LIMIT - hlt.networking.Game.calculate_distance(pos, goal)):
        # logging.debug(f"neighbors of {pos}: {list(pos.neighbors(map))}")
        return pos.neighbors(map)
    else:
        return []
neighbors(map, home, ship.position)

def dist_home(pos, next):
    # logging.debug('dist_home')
    d = hlt.networking.Game.calculate_distance(pos, next)
    d *= MAX_HALITE / 28.5
    # logging.debug(f"dist between {pos}->{next} = {d}")
    return d

def step_cost(pos, next):
    return MAX_HALITE / 80

def gain(pos, next):
    return next.ship_halite - pos.ship_halite

def move_heur(pos, next):
    g = gain(pos, next)
    c = step_cost(pos, next)

    # logging.debug(f"cost {pos}->{next} = {c} - {g} = {c-g}")
    checks[next] += 1
    return c-g

# def goal_reached(pos, goal):
#     return pos.x == goal.x and pos.y == goal.y

pather = MultiagentAStar()
pather.neighbors = lambda *x: neighbors(map, home, *x)
# pather.is_goal_reached = goal_reached

def log_checks(ret, pos, next):
    checks[next] += 1
    return ret

def log_path(name, path):
    halite = path[ship.position][-1].ship_halite
    unique = {Position(n) for n in path[ship.position]}
    logging.info(f"{name}\nnodes {len(unique)} turns {len(path[ship.position])} checks {int(np.sum(checks)):04} halite {int(halite)}")
    logging.info(path[ship.position])

def draw_path(img, path):
    if path[0] is not None:
        logging.debug(f"draw {path}")
        for node in path:
            logging.debug(f"draw {node}")
            img[node] += 1.0

# paths[ship.position] += 0.5
# paths[home] += 0.5
points[ship.position] += 1.0
points[home] += 1.0

cv2.namedWindow('halite', cv2.WINDOW_NORMAL)
cv2.resizeWindow('halite', 500, 500)
stack = np.dstack([points, map/MAX_HALITE, np.zeros_like(map)])
cv2.imshow('halite', stack)  # bgr
cv2.setMouseCallback('halite', lambda *x: onmouse('halite', stack, *x))


# # astar
# pather.distance_between = move_heur
# pather.heuristic_cost_estimate = dist_home
# checks = np.zeros_like(map)
# apath = pather.astar([ship.position], home)
# log_path('astar', apath)
# apath_img = paths.copy()
# draw_path(apath_img, apath[ship.position])
# cv2.namedWindow('apath', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('apath', 500, 500)
# stack = np.dstack([apath_img, visited, checks/np.amax(checks)])
# cv2.imshow('apath', stack)  # bgr
# cv2.setMouseCallback('apath', lambda *x: onmouse('apath', np.dstack([apath_img, visited, checks]), *x))

# dijkstras
# pather.distance_between = lambda *x: -1*gain(*x)
pather.distance_between = step_cost
pather.heuristic_cost_estimate = lambda *x: 0
checks = np.zeros_like(map)
dpath = pather.astar([ship.position], home)
log_path('dijkstras', dpath)
dpath_img = paths.copy()
draw_path(dpath_img, dpath[ship.position])
cv2.namedWindow('dpath', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dpath', 500, 500)
stack = np.dstack([dpath_img, visited, checks/np.amax(checks)])
cv2.imshow('dpath', stack)  # bgr
cv2.setMouseCallback('dpath', lambda *x: onmouse('dpath', np.dstack([stack[:,:,0], stack[:,:,1], checks]), *x))

# # bfs
# pather.distance_between = lambda *x: log_checks(1, *x)
# pather.heuristic_cost_estimate = lambda *x: 0
# checks = np.zeros_like(map)
# bpath = pather.astar([ship.position], home)
# log_path('bfs', bpath)
# bpath_img = paths.copy()
# draw_path(bpath_img, bpath[ship.position])
# cv2.namedWindow('bpath', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('bpath', 500, 500)
# stack = np.dstack([bpath_img, visited, checks/np.amax(checks)])
# cv2.imshow('bpath', stack)  # bgr
# cv2.setMouseCallback('bpath', lambda *x: onmouse('bpath', np.dstack([stack[:,:,0], stack[:,:,1], checks]), *x))
#
# # greedy
# pather.distance_between = lambda *x: log_checks(15000, *x)
# pather.heuristic_cost_estimate = dist_home
# checks = np.zeros_like(map)
# gpath = pather.astar([ship.position], home)
# log_path('greedy', gpath)
# gpath_img = paths.copy()
# draw_path(gpath_img, gpath[ship.position])
# cv2.namedWindow('gpath', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('gpath', 500, 500)
# stack = np.dstack([gpath_img, visited, checks/np.amax(checks)])
# cv2.imshow('gpath', stack)  # bgr
# cv2.setMouseCallback('gpath', lambda *x: onmouse('gpath', np.dstack([stack[:,:,0], stack[:,:,1], checks]), *x))

cv2.waitKey(0)
cv2.destroyAllWindows()
