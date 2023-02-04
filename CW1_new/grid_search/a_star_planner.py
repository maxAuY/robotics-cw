'''
Created on 2 Jan 2022

@author: ucacsjj
'''
from math import sqrt
from .dijkstra_planner import DijkstraPlanner
from heapq import heapify

# This class implements the A* search algorithm

from enum import Enum

class AStarPlanner(DijkstraPlanner):
    
    def __init__(self, occupancy_grid):
        DijkstraPlanner.__init__(self, occupancy_grid)

    # Q2d:
    # Complete implementation of A*.

    def push_cell_onto_queue(self, cell):
        # calculate path cost
        parent = cell.parent
        if parent == None:
            cell.path_cost = 0
        else:
            transition_cost = self.compute_l_stage_additive_cost(parent, cell)
            cell.path_cost = transition_cost + parent.path_cost
        # calculate heuristic
        heuristic = self.calc_heuristic(cell)
        # add the two and use for determining priority
        self.priority_queue.put((cell.path_cost+heuristic,cell))

    def calc_heuristic(self,cell):
        # Euclidean distance to the goal
        cell_coords = cell.coords()
        goal_coords = self.goal.coords()

        dX = cell_coords[0] - goal_coords[0]
        dY = cell_coords[1] - goal_coords[1]
        d = sqrt(dX * dX + dY * dY)
        
        return d

    def resolve_duplicate(self, cell, parent_cell):
        transition_cost = self.compute_l_stage_additive_cost(parent_cell, cell)
        new_path_cost = parent_cell.path_cost + transition_cost

        if new_path_cost < cell.path_cost:
            queue = self.priority_queue.queue

            cell.path_cost = new_path_cost
            cell.set_parent(parent_cell)

            # update queue
            for i in range(self.priority_queue._qsize()):
                # remove old entry
                cell_q = self.priority_queue.queue[i]
                if cell_q[1] is cell:
                    del queue[i]
                    heapify(queue)
                    break

            # add new entry
            heuristic = self.calc_heuristic(cell)
            self.priority_queue.put((cell.path_cost+heuristic,cell))