from astar import AStar, find_path
from types import GeneratorType
from cachetools import LFUCache, LRUCache


def cached_find_path(start, goal, neighbors_fnct, reversePath=False, heuristic_cost_estimate_fnct=lambda a, b: Infinite, distance_between_fnct=lambda a, b: 1.0, is_goal_reached_fnct=lambda a, b: a == b, maxsize=64*64):
    """A non-class version of the path finding algorithm"""
    class FindPath(AStar):
        @cached(LFUCache(maxsize=maxsize))
        def heuristic_cost_estimate(self, current, goal):
            return heuristic_cost_estimate_fnct(current, goal)
        @cached(LFUCache(maxsize=maxsize))
        def distance_between(self, n1, n2):
            return distance_between_fnct(n1, n2)
        @cached(LFUCache(maxsize=maxsize))
        def neighbors(self, node):
            return neighbors_fnct(node)
        @cached(LFUCache(maxsize=maxsize))
        def is_goal_reached(self, current, goal):
            return is_goal_reached_fnct(current, goal)
        def clear_cache(self):
            self.heuristic_cost_estimate.cache_clear()
            self.distance_between.cache_clear()
            self.neighbors.cache_clear()
            self.is_goal_reached.cache_clear()
    return FindPath().astar(start, goal, reversePath)


def astar_multiagent(starts, *args, **kwargs):
    """Applies A* algorithm to multiple agents/start points.
    Uses python-astar, caching paths as they are discovered,
    for use by all agents in 'starts'.
    
    Starts can be a single object or iterable of objects.
    *Individual objects must be hashable.
    args, kwargs are same as 'astar()'
    
    returns dict, where k=starting node, v=[nodes along path to goal]
    """
    routes = {}  # dict, k=starting node, v=path to goal
    
    # for each starting node/agent
    for node in starts:
        # if we don't have cached path yet
        if node not in routes.keys():
            # find optimal path
            path = list(find_path(node, *args, **kwargs) or [None])
            
            if path == [None]:
                routes[node] = [None]
                continue
            
            #if kwargs.get('reversePath', False):
            #    path = path[::-1]
            
            # cache explored paths
            for idx, _node in enumerate(path):
                #if _node in starts:
                #routes[_node] = path[idx:] if not kwargs.get('reversePath',False) else path[idx:][::-1]
                routes[_node] = path[idx:].reverse() if not kwargs.get('reversePath',False) else path[idx:]
    return routes

def astar_multiagent_no_cache(starts, *args, **kwargs):
    """Applies A* algorithm to multiple agents/start points.
    Uses python-astar as backend.
    
    Starts can be a single object or iterable of objects.
    *Individual objects must be hashable.
    args, kwargs are same as 'astar()'
    
    returns dict, where k=starting node, v=[nodes along path to goal]
    """
    routes = {}  # dict, k=starting node, v=path to goal
    
    # for each starting node/agent
    for node in starts:
        # find optimal path
        path = list(find_path(node, *args, **kwargs))
        routes[node] = path
    return routes

def astar(starts, *args, **kwargs):
    """Applies A* algorithm to multiple agents/start points.
    Uses python-astar, caching paths as they are discovered,
    for use by all agents in 'starts'.
    
    Starts can be a single object or iterable of objects.
    *Individual objects must be hashable.
    
    args, kwargs are same as python-astar's find_path()
        goal:
            Typically the goal node. Can be just about anything.
            Second value passed to is_goal_reached_fnct.
            Second value passed to heuristic_cost_estimate_fnct.
        neighbors_fnct:
            Returns an iterable of nodes reachable from the current node
            params: current node being invesigated
        reversePath=False:
            bool, whether to reverse returned path order
        distance_between_fnct:
            Calculates distance estimate (travel cost) to next node.
            Defaults to ::  1
            params: current node being investigated,
                    connected node to check distance to
        heuristic_cost_estimate_fnct:
            Calculates distance estimate to goal node.
            Defaults to ::  Infinite
            params: current node being investigated,
                    passed in 'goal'
        is_goal_reached_fnct:
            Evaluates to True when the current node is identified as
            a goal, False otherwise.
            Defaults to ::  ( current_node==goal )
            params: current node being investigated,
                    passed in 'goal'
    
    returns dict, where k=starting node, v=[nodes along path to goal]
    """
    #if hasattr(starts, '__iter__') and not isinstance(starts, tuple):
    if isinstance(starts, (list, GeneratorType)):
        return astar_multiagent(starts, *args, **kwargs)
    return {starts:list(find_path(starts, *args, **kwargs) or [None])}

def dijkstras(starts, *args, **kwargs):
    """Applies Dijkstras algorithm to multiple agents/start points.
    Uses python-astar, caching paths as they are discovered,
    for use by all agents in 'starts'.
    
    Starts can be a single object or iterable of objects.
    *Individual objects must be hashable.
    args, kwargs are same as 'astar()'
    
    Note: Dijkstras == A* where heuristic is ignored (== 0)
    
    returns dict, where k=starting node, v=[nodes along path to goal]
    """
    assert 'heuristic_cost_estimate_fnct' not in kwargs,\
        "Dijkstras algorithm does not use a heuristic (heuristic_cost_estimate_fnct)"
    #if hasattr(starts, '__iter__') and not isinstance(starts, tuple):
    if isinstance(starts, (list, GeneratorType)):
        return astar_multiagent(starts, *args, heuristic_cost_estimate_fnct=lambda *x:0, **kwargs)
    return astar(starts, *args, heuristic_cost_estimate_fnct=lambda *x:0, **kwargs)

def greedy(starts, *args, **kwargs):
    """Applies a greedy best-first algorithm to multiple agents/start points.
    Uses python-astar, caching paths as they are discovered,
    for use by all agents in 'starts'.
    
    Starts can be a single object or iterable of objects.
    *Individual objects must be hashable.
    args, kwargs are same as 'astar()'
    
    Note: greedy == A* where travel distance between nodes is ignored (== 0)
    
    returns dict, where k=starting node, v=[nodes along path to goal]
    """
    assert 'distance_between_fnct' not in kwargs,\
        "greedy algorithm does not use a travel cost (distance_between_fnct)"
    #if hasattr(starts, '__iter__') and not isinstance(starts, tuple):
    if isinstance(starts, (list, GeneratorType)):
        return astar_multiagent(starts, *args, distance_between_fnct=lambda *x:0, **kwargs)
    return astar(starts, *args, distance_between_fnct=lambda *x:0, **kwargs)


__all__ = ['astar','dijkstras','greedy','cached_find_path','find_path']

