#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario

from p1.high_level_environment import PlannerType
from p1.high_level_environment import HighLevelEnvironment
from p1.high_level_actions import HighLevelActionType



if __name__ == '__main__':
    
    # Create the scenario
    airport_map, drawer_height = full_scenario()
    
    # Just use Euclidean distance for traversability costs for now
    airport_map.set_use_cell_type_traversability_costs(False)
    
    # Draw what the map looks like. This is optional and you
    # can comment it out
    # airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    # airport_map_drawer.update()    
    # airport_map_drawer.wait_for_key_press()
        
    # Create the gym environment
    # Q1e:
    # You will need to enable your implementation of Dijkstra
    airport_environment = HighLevelEnvironment(airport_map, PlannerType.DIJKSTRA)
    
    # Show the graphics
    airport_environment.show_graphics(True)
    airport_environment.show_verbose_graphics(False)
    
    # First specify the start location of the robot
    action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    if reward is -float('inf'):
        print('Unable to teleport to (1, 1)')
        
    # Get all the rubbish bins and toilets; these are places which need cleaning
    all_rubbish_bins = airport_map.all_rubbish_bins()
        
    # Q1e:
    # Modify to collect statistics for assessing algorithms
    # Now go through them and plan a path sequentially

    total_cells_visited = 0
    total_cost = 0

    for rubbish_bin in all_rubbish_bins:
        action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
        observation, reward, done, info = airport_environment.step(action)
        
        total_cells_visited += info.number_of_cells_visited
        total_cost -= reward 
            
        try:
            input("Press enter in the command window to continue.....")
        except SyntaxError:
            pass  
     
    print('total cost is: ',total_cost)
    print('total cells visited is: ',total_cells_visited)   
