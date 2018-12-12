from astar import AStar


def dijkstras(self, node, goal, reversePath=False):
    h = self.heuristic_cost_estimate
    self.heuristic_cost_estimate = lambda *x: 0
    result = self.astar(node, goal, reversePath)
    self.heuristic_cost_estimate = h
    return result


def greedy(self, node, goal, reversePath=False):
    d = self.distance_between
    self.distance_between = lambda *x: 0
    result = self.astar(node, goal, reversePath)
    self.distance_between = d
    return result

def bfs(self, node, goal, reversePath=False):
    d = self.distance_between
    self.distance_between = lambda *x: 1
    h = self.heuristic_cost_estimate
    self.heuristic_cost_estimate = lambda *x: 0

    result = self.astar(node, goal, reversePath)

    self.distance_between = d
    self.heuristic_cost_estimate = h
    return result


setattr(AStar, 'dijkstras', dijkstras)
setattr(AStar, 'greedy', greedy)
setattr(AStar, 'bfs', bfs)


class MultiagentAStar(AStar):
    def astar(self, starts, goal, reversePath=False):
        routes = {}
        for start in starts:
            routes[start] = list(super().astar(start, goal, reversePath))
        return routes


class CachedAStar(AStar):
    """Applies A* algorithm to multiple agents/start points.
        Uses python-astar, caching paths as they are discovered,
        for use by all agents in 'starts'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routes = {}  # dict, k=starting node, v=path to goal

    def reset_cache(self, nodes=None):
        """Remove cached paths for specified nodes (default is all nodes).

        :param nodes: iterable of nodes to remove from cache
        :return: nothing
        """
        if nodes is None:
            self.routes.clear()
        else:
            for node in nodes:
                self.routes.pop(node, None)

    def astar(self, nodes, goal, reversePath=False):
        """

        :param nodes: starting points to pathfind to goal
        :param goal: 2nd param passed to is_goal_reached,
                also passed to distance_between for initial sort
        :param reversePath: whether to reverse the returned paths
        :return: {starting_node:[nodes to goal], ... }
        """
        # sort nodes by distance from goal
        # could save a lot with many nodes, assuming distance process cost is small
        # logging.debug("START SORTING")
        # nodes = sorted(nodes, key=lambda x: self.distance_between(x, goal))
        # logging.debug("DONE SORTING")

        # for each starting node/agent
        for node in nodes:
            # if we don't have cached path yet
            if node not in self.routes.keys():
                # find optimal path
                path = list(super().astar(node, goal, reversePath) or [])

                if path == []:
                    self.routes[node] = []
                    continue

                # cache explored paths
                for idx, _node in enumerate(path):
                    self.routes[_node] = path[-1:idx:-1] if not reversePath else path[idx:]
        return self.routes


__all__ = ['AStar', 'MultiagentAStar', 'CachedAStar']
