from queue import Queue
import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from shapely.geometry import LineString, Point


class Node:
    def __init__(self, id, parent, accumulated_cost=0, estimated_cost=0):
        self.id = id
        self.parent = parent
        # accumulated costs = the costs to walk from the start node to the current node
        self.accumulated_cost = accumulated_cost
        # estimated costs = the costs based on the heuristic
        self.estimated_cost = estimated_cost
        self.total_estimated_cost = self.accumulated_cost + self.estimated_cost

    def __lt__(self, other):
        return self.total_estimated_cost < other.total_estimated_cost

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.id)
    
class SearchAlgorithm:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.fig = None
        self.ax = None

    def search(self):
        pass

    def reconstruct_path(self, goal_node):
        # Construct the path from the goal node
        shortest_path = []
        current_node = goal_node
        while current_node.parent is not None:
            shortest_path.insert(0, (current_node.parent.id, current_node.id))
            current_node = current_node.parent
        return shortest_path
    

class BFS(SearchAlgorithm):
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)
        self.open_list = Queue()
        self.closed_list = set()
        self.path = []
        self.start_node = Node(start, None)
        self.open_list.put(self.start_node)
        self.goal_node = Node(goal, None)
        self.algorithm_name = "BFS"

    def search(self):
        while not self.open_list.empty():
            current_node = self.open_list.get()


            if current_node.id == self.goal_node.id:
                # If the goal is reached, update the shortest path
                self.shortest_path = self.reconstruct_path(current_node)
                return self.shortest_path
            
            
            self.closed_list.add(current_node)

            current_step = []
            for neighbor, data in self.graph[current_node.id].items():
                neighbor_node = Node(neighbor, current_node, accumulated_cost=current_node.accumulated_cost + data.get("length", 1))
                neighbor_node.total_estimated_cost = neighbor_node.accumulated_cost

                if neighbor_node in self.closed_list:
                    continue
                if neighbor_node not in self.open_list.queue:
                    self.open_list.put(neighbor_node)

                current_step.append((current_node.id, neighbor_node.id))

            if current_step:
                self.path.append(current_step)  # Add the current step to the path


        return None
    
class DFS(SearchAlgorithm):
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)
        self.start_node = Node(start, None)
        self.goal_node = Node(goal, None)
        self.stack = []
        self.visited = set()
        self.path = []
        self.algorithm_name = "DFS"

    def search(self):
        self.stack.append(self.start_node)

        while self.stack:
            current_node = self.stack.pop()

            if current_node.id == self.goal_node.id:
                self.shortest_path = self.reconstruct_path(current_node)
                return self.shortest_path

            if current_node.id not in self.visited:
                self.visited.add(current_node.id)

                current_step = []
                for neighbor, data in self.graph[current_node.id].items():
                    neighbor_node = Node(neighbor, current_node)

                    if neighbor_node.id not in self.visited:
                        self.stack.append(neighbor_node)
                        current_step.append((current_node.id, neighbor_node.id))

                if current_step:
                    self.path.append(current_step)

        return None
    
class Greedy(SearchAlgorithm):
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)
        self.open_list = []
        self.closed_list = set()
        self.path = []
        self.start_node = Node(start, None)
        heapq.heappush(self.open_list, self.start_node)
        self.goal_node = Node(goal, None)
        self.algorithm_name = "Greedy"

    def heuristic(self, current_node, goal_node):
        """Uses the great circle distance as the heuristic to estimate the distance between the current node and the goal node.

        Args:
            current_node (int): the integer id of the current node
            goal_node (int): the integer id of the goal node

        Returns:
            float: the estimated distance between the current node and the goal node
        """
        return ox.distance.great_circle(self.graph.nodes[current_node]['y'], self.graph.nodes[current_node]['x'], self.graph.nodes[goal_node]['y'], self.graph.nodes[goal_node]['x'])

    def search(self):
        while self.open_list:
            current_node = heapq.heappop(self.open_list)

            if current_node.id == self.goal_node.id:
                self.shortest_path = self.reconstruct_path(current_node)
                return self.shortest_path
            
            self.closed_list.add(current_node)

            current_step = []
            for neighbor, data in self.graph[current_node.id].items():
                neighbor_node = Node(neighbor, current_node)

                if neighbor_node in self.closed_list:
                    continue

                if neighbor_node not in self.open_list:
                    neighbor_node.estimated_cost = self.heuristic(neighbor, self.goal_node.id)
                    neighbor_node.total_estimated_cost = neighbor_node.estimated_cost
                    heapq.heappush(self.open_list, neighbor_node)

                # to check if the current path is better than the previous one
                elif neighbor_node in self.open_list:
                    
                    node = self.open_list[self.open_list.index(neighbor_node)]
                    
                    if node.total_estimated_cost > neighbor_node.total_estimated_cost:
                        # remove the old node from the open list
                        self.open_list.remove(node)

                        node.estimated_cost = neighbor_node.estimated_cost
                        node.total_estimated_cost = neighbor_node.total_estimated_cost
                        node.parent = neighbor_node.parent

                        # add the updated node to the open list
                        heapq.heappush(self.open_list, node)
                
                current_step.append((current_node.id, neighbor_node.id))

            if current_step:
                self.path.append(current_step)
        return None

class AStar(SearchAlgorithm):
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)
        self.open_list = list()
        self.closed_list = set()
        self.path = []
        self.start_node = Node(start, None)
        heapq.heappush(self.open_list, self.start_node)

        # not sure if costs should be initialized to 0
        self.goal_node = Node(goal, None, accumulated_cost=0, estimated_cost=self.heuristic(start, goal))
        self.algorithm_name = "AStar"

    def heuristic(self, current_node, goal_node):
        """Uses the great circle distance as the heuristic to estimate the distance between the current node and the goal node.

        Args:
            current_node (int): the integer id of the current node
            goal_node (int): the integer id of the goal node

        Returns:
            float: the estimated distance between the current node and the goal node
        """
        return ox.distance.great_circle(self.graph.nodes[current_node]['y'], self.graph.nodes[current_node]['x'], self.graph.nodes[goal_node]['y'], self.graph.nodes[goal_node]['x'])

    def search(self):
        while self.open_list:
            current_node = heapq.heappop(self.open_list)

            if current_node.id == self.goal_node.id:
                self.shortest_path = self.reconstruct_path(current_node)
                return self.shortest_path
            
            self.closed_list.add(current_node)

            current_step = []
            for neighbor, data in self.graph[current_node.id].items():
                # TODO: refactor this to a function
                if not data[0].get('length'):
                    raise Exception("No length attribute found for edge: ", current_node.id, neighbor) 

                neighbor_node = Node(neighbor, current_node)

                if neighbor_node in self.closed_list:
                    continue

                if neighbor_node not in self.open_list:
                    neighbor_node.accumulated_cost = current_node.accumulated_cost + data[0].get('length')
                    neighbor_node.estimated_cost = self.heuristic(neighbor, self.goal_node.id)
                    heapq.heappush(self.open_list, neighbor_node)

                # to check if the current path is better than the previous one
                elif neighbor_node in self.open_list:
                    
                    node = self.open_list[self.open_list.index(neighbor_node)]
                    
                    if node.total_estimated_cost > neighbor_node.total_estimated_cost:
                        # remove the old node from the open list
                        self.open_list.remove(node)

                        node.accumulated_cost = neighbor_node.accumulated_cost
                        node.estimated_cost = neighbor_node.estimated_cost
                        node.total_estimated_cost = neighbor_node.total_estimated_cost
                        node.parent = neighbor_node.parent

                        # add the updated node to the open list
                        heapq.heappush(self.open_list, node)
                
                current_step.append((current_node.id, neighbor_node.id))

            if current_step:
                self.path.append(current_step)
        return "No solution found"