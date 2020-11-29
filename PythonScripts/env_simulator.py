# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:12:43 2020

@author: jjling.2018
"""

import numpy as np

class MakeEnv:
    def __init__(self, grid_map, num_agents, starts, goals):
        self.map = grid_map
        self.num_agents = num_agents
        self.num_zones = list(self.map.keys())
        self.starts = starts
        self.goals = goals
        self.num_agents_zone = {}
        self.local_done = [False for _ in range(self.num_agents)]
        self.vector_observations = np.ones((self.num_agents, 15), dtype=np.int8) * -1
        self.rewards = [np.nan for _ in range(self.num_agents)]
        
    def reset(self):
        for nn in self.num_zones:
            self.num_agents_zone[nn] = 0
        for start_zone in self.starts:
            self.num_agents_zone[start_zone] += 1
        for ii in range(self.num_agents):
            ob = []
            ob = [ii] + [self.starts[ii]] + [-1,-1] + [self.goals[ii]] + [self.starts[ii]] + [self.num_agents_zone[self.starts[ii]]]
            for nn in self.map[self.starts[ii]]:
                ob += [nn] + [self.num_agents_zone[nn]]             
            self.vector_observations[ii, 0:len(ob)] = ob
        
    def step(self, actions):
        #reward and update state 
        for ii in range(self.num_agents):
            current_zone = self.vector_observations[ii, 1]
            current_zone_copy = self.vector_observations[ii, 1]
            next_zone = self.vector_observations[ii, 2]
            time_to_zone = self.vector_observations[ii, 3]
            goal = self.vector_observations[ii, 4]
            goal_copy = self.vector_observations[ii, 4]

            dis_act = actions[ii, 0]
            con_act = actions[ii, 1]
            
            if current_zone != goal and goal != -1:
                if dis_act == -1:
                    if time_to_zone == 1:
                        self.vector_observations[ii, 1] = next_zone
                        self.vector_observations[ii, 2] = -1
                        self.vector_observations[ii, 3] = -1
                        self.num_agents_zone[current_zone] -= 1
                        self.num_agents_zone[next_zone] += 1 
                    else:
                        self.vector_observations[ii, 3] -= 1
                else:
                    if con_act == 1:
                        self.vector_observations[ii, 1] = dis_act
                        self.vector_observations[ii, 2] = -1
                        self.vector_observations[ii, 3] = -1
                        self.num_agents_zone[current_zone] -= 1
                        self.num_agents_zone[dis_act] += 1 
                    else:
                        self.vector_observations[ii, 2] = dis_act
                        self.vector_observations[ii, 3] = con_act-1
            else:
                if current_zone == goal:
                    self.vector_observations[ii, 4] = -1
                    
            #update reward and local done
            if current_zone_copy == goal_copy:
                self.local_done[ii] = True
                self.rewards[ii] = 100
            else:
                self.local_done[ii] = False
                if goal == -1:
                    self.rewards[ii] = 0
                else:
                    self.rewards[ii] = -1

        #update count
        for ii in range(self.num_agents):
            counting = []
            current_zone = self.vector_observations[ii, 1]
            counting  = [current_zone] + [self.num_agents_zone[current_zone]]
            for nn in self.map[current_zone]:
                counting += [nn] + [self.num_agents_zone[nn]]             
            self.vector_observations[ii, 5:] = counting + [-1]*(10-len(counting))
        
    def close(self):
         return True
        

        
        
        