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
    
    # Q1f:
    # Enable using cell type dependent traversability
    # costs. This is the default, and will not
    # explicitly be called in other files.
    airport_map.set_use_cell_type_traversability_costs(True)
    
    # Draw what the map looks like. This is optional and you
    # can comment it out
    # drawer_height = 500
    # airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    # airport_map_drawer.update()    
    # airport_map_drawer.wait_for_key_press()
        
    # Create the gym environment
    # Q1d:
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
        
    # Q1f:
    # Modify to collect statistics for assessing algorithms
    # Now go through them and plan a path sequentially

    bin_number = 1
    total_cells_visited = 0
    total_cost = 0
    all_path_costs = []  
    all_cells_visited = []

    for rubbish_bin in all_rubbish_bins:
        action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
        observation, reward, done, info = airport_environment.step(action)
        
        total_cells_visited += info.number_of_cells_visited
        total_cost -= reward 

        all_path_costs.append((bin_number,-reward))
        all_cells_visited.append((bin_number,info.number_of_cells_visited))

        screen_shot_name = f'bin_{bin_number:02}.pdf'
        # airport_environment.search_grid_drawer().save_screenshot(screen_shot_name)
        
        bin_number += 1
        try:
            input("Press enter in the command window to continue.....")
        except SyntaxError:
            pass  
     
    print('total cost is: ',total_cost)
    print('total cells visited is: ',total_cells_visited)   
    print('all path costs: ', all_path_costs)
    print('all cells visited: ', all_cells_visited)
    
