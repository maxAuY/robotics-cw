#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import *

from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

import time

if __name__ == '__main__':
    
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    drawer_height = 300
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(.8)
    
    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(5)
    # reduce maximum number of iterations to reduce time allocated
    # policy_solver.set_max_policy_iteration_steps(13)
    
    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)

    # start timing the policy iterator
    start_time = time.time()    

    # Compute the solution
    v1, pi1 = policy_solver.solve_policy()

    # stop timing policy iterator
    end_time = time.time()
    policy_time = end_time-start_time
    
    # # Save screen shot; this is in the current directory
    # policy_drawer.save_screenshot("policy_iteration_results.pdf")
    # value_function_drawer.save_screenshot('policy_iteration_values.pdf')
    
    # Q3i: Add code to evaluate value iteration down here.
    
    # Create the policy iterator
    policy_solver = ValueIterator(airport_environment)
    # reduce max number of iterations for limiting time
    # policy_solver.set_max_optimal_value_function_iterations(57)
    
    # Set up initial state
    policy_solver.initialize()
    
    # Bind the drawer with the solver
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)

    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
    
    # start timing the policy iterator
    start_time = time.time()    
    
    # Compute the solution
    v2, pi2 = policy_solver.solve_policy()

    # stop timing
    end_time = time.time()
    value_time = end_time-start_time
    
    # Save screen shot; this is in the current directory
    # policy_drawer.save_screenshot("value_iteration_results.pdf")
    # value_function_drawer.save_screenshot("value_iteration_values.pdf")

    print('total time for policy iterator: ', policy_time)
    print('total time for value iterator: ', value_time)

    policy_drawer.wait_for_key_press()