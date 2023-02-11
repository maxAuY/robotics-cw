'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from queue import PriorityQueue
from heapq import heapify

from .planner_base import PlannerBase

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancy_grid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()

    # Q1d:
    # Modify this class to finish implementing Dijkstra
    # path cost into queue
    def push_cell_onto_queue(self, cell):
        parent = cell.parent
        if parent == None:
            cell.path_cost = 0
        else:
            transition_cost = self.compute_l_stage_additive_cost(parent, cell)
            cell.path_cost = transition_cost + parent.path_cost

        self.priority_queue.put((cell.path_cost,cell))

    # Check the queue size is zero
    def is_queue_empty(self):
        return self.priority_queue.empty()

    # Simply pull from the front of the list
    def pop_cell_from_queue(self):
        cell = self.priority_queue._get()
        return cell[1]

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
                cell_q = queue[i]
                if cell_q[1] is cell:
                    del queue[i]
                    heapify(queue)
                    break

            # add new entry
            self.priority_queue.put((cell.path_cost,cell))


