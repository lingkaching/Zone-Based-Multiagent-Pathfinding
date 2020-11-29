# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:23:27 2019

@author: jjling.2018
"""
import json
from itertools import groupby
import tensorflow as tf
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import sys
import random
import time
from collections import defaultdict
from collections import deque
from mlagents.envs import UnityEnvironment
from optparse import OptionParser
import os
#Change #4:
from pathos.multiprocessing import ThreadingPool as Pool

class Actor:
	def __init__(self, bi_n, learningRate, num_of_GOALS, input_var, input_var1, numUnitsPerHLayer, neighbours, zoneID = None):
		self.num_of_GOALS = num_of_GOALS
		self.bi_n = bi_n
		self.zoneID = zoneID
		self.input_var = input_var
		self.input_var1 = input_var1
		self.numUnitsPerHLayer = numUnitsPerHLayer
		self.neighbours = neighbours    
		with tf.variable_scope('Actor'):   
			self.output_var = self.initPolNN(scope='eval', trainable=True)

		
		#action
		self.act_var = None
		#reward
		self.ret_var = None
		self.nextret_var = None 
		self.phi = None
		self.selected_dis_action = None
		self.selected_con_action = None
		self.returnAndSelectedPolicy = None
		self.zoneBasedVal = None
		self.finalObj = None
		self.val1 = None
		self.val2 = None
		self.val3 = None
		self.learning_rate = learningRate
		self.optimizer = None
		self.learning_step = None
		self.gamma = 0.99
		
	def setCompGraph(self):
		with tf.variable_scope('ActorTraning'):
			self.act_var = tf.placeholder(shape=[None, self.num_of_GOALS, self.neighbours], dtype=tf.float32)
			self.ret_var = tf.placeholder(shape=[1,None], dtype=tf.float32)
			self.nextret_var = tf.placeholder(shape=[1,None], dtype=tf.float32)
			self.phi = tf.placeholder(shape=[1,None], dtype=tf.float32)
		
			temp1 = 0
			temp2 = 0
			for nn in range(self.num_of_GOALS):
				temp1 = tf.add(self.output_var[nn] * self.act_var[:,nn], temp1)
			
				temp2 = tf.add(self.output_var[nn+self.num_of_GOALS] * tf.reduce_sum(self.act_var[:,nn],axis=1,keepdims=True), temp2)
			
			self.selected_dis_action = tf.reduce_sum(temp1, axis=1)                      
			
			self.selected_con_action = tf.reduce_sum(temp2, axis=1)
						
			self.val1 = tf.log(self.selected_dis_action + 1e-8) * self.ret_var

			self.val2 = tf.add(tf.log(self.selected_con_action + 1e-8 ),tf.negative(tf.log1p(1e-8 + tf.negative(self.selected_con_action))))* (self.nextret_var * self.phi)
	 
			self.val3 = tf.log1p(1e-8 + tf.negative(self.selected_con_action))  * (self.nextret_var *  self.bi_n)
			
			self.zoneBasedVal = tf.add(self.val1, self.gamma*(tf.add(self.val2, self.val3)))
			
			self.finalObj = tf.reduce_sum(self.zoneBasedVal, axis=1) 

			self.loss = -1*(self.finalObj)

			# Learning Rate
			# self.learning_rate = 1e-4

			# Defining Optimizer
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

			# Update Gradients
			self.learning_step = (self.optimizer.minimize(self.loss))

	def initPolNN(self, scope, trainable):
		OUTPUT = {}
		with tf.variable_scope(scope):       
			l_in = tf.layers.dense(inputs=self.input_var, units=self.numUnitsPerHLayer, activation=tf.nn.relu, trainable=trainable, name=str(self.zoneID)+"-DenseReLu_1")           
			
			l_in_norm = tf.contrib.layers.layer_norm(inputs=l_in, trainable=trainable, scope=str(self.zoneID)+'norm1')           
			
			l_hid_1 = tf.layers.dense(inputs=l_in_norm, units=self.numUnitsPerHLayer, activation=tf.nn.relu, trainable=trainable, name=str(self.zoneID)+"-DenseReLu_2")           
			
			l_hid_1_norm = tf.contrib.layers.layer_norm(inputs=l_hid_1, trainable=trainable, scope=str(self.zoneID)+'norm2')           
			
			l_hid_2 = tf.layers.dense(inputs=l_hid_1_norm, units=self.numUnitsPerHLayer, activation=tf.nn.relu, trainable=trainable, name=str(self.zoneID)+"-DenseReLu_3")                   
			
			l_hid_2_norm = tf.contrib.layers.layer_norm(inputs=l_hid_2, trainable=trainable, scope=str(self.zoneID)+'norm3')
			
			
			for nn in range(self.num_of_GOALS):
			
				l_discrete_out = tf.layers.dense(inputs=l_hid_2_norm, units=self.neighbours, activation=tf.nn.softmax, trainable=trainable, name=str(self.zoneID)+"-Softmax"+str(nn))        
			
				l_continuous_out = tf.layers.dense(inputs=tf.concat([self.input_var1[:,nn], l_hid_2_norm], 1), units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(), trainable=trainable, name=str(self.zoneID)+"-Sigmoid"+str(nn))               
			
				OUTPUT[nn] = l_discrete_out
			
				OUTPUT[nn+self.num_of_GOALS] = l_continuous_out
			
			return OUTPUT

	def getAction(self, sess, GoalObs, DisActions):
		return sess.run(self.output_var, feed_dict={self.input_var : GoalObs, self.input_var1 : DisActions})
		   
def shortestPath(modelPath):

	f = open(modelPath, 'r')
	model = f.read().split('\n')
	#all pair shortest path
	edges = []
	for i in range(3, len(model)):
		edgeRecords = model[i].split(' ')[:-1]
		for j in range(1, len(edgeRecords)):
			edges.append((edgeRecords[0], edgeRecords[j]))  
	#create directed graph
	G = nx.DiGraph()
	G.add_edges_from(edges)    
	length = dict(nx.all_pairs_shortest_path_length(G))
	return length

def ConstructGraph(modelPath):
	f = open(modelPath, 'r')
	model = f.read().split('\n')
	num_of_zones = int(model[0].strip().split(':')[-1].strip())
	edges = {}
	for i in range(3, len(model)):
		if(len(model[i]) == 0):
			continue
		edgeRecords = model[i].strip().split(' ')
#         edges[int(edgeRecords[0])] = [int(edgeRecords[0])] + [int(x) for x in edgeRecords[1:]]
		edges[int(edgeRecords[0])] = [int(x) for x in edgeRecords[1:]]      
	return edges

def InitialiseActor(g_1, lr, edges, num_of_zones, num_of_agents, GOALS, T_min, T_max):
	num_of_GOALS = len(np.unique(GOALS))
	with g_1.as_default():
		polNNs = []
		#input_vars = []
		for i in range(0, num_of_zones):
			inp_var = tf.placeholder(shape=[None, num_of_zones+num_of_agents], dtype=tf.float32)
			inp_var1 = tf.placeholder(shape=[None, num_of_GOALS, len(edges[i])], dtype=tf.float32)
			#input_vars.append(inp_var) 
			polNN = Actor(bi_n = T_max-T_min, learningRate = lr, num_of_GOALS=num_of_GOALS, input_var=inp_var, input_var1=inp_var1, numUnitsPerHLayer=64, neighbours=len(edges[i]), zoneID=str(i))
			polNN.setCompGraph()
			polNNs.append(polNN)
	return polNNs


def loadUnityEnvironment(options):
	#print("Python version:")
	#print(sys.version)
	# check Python version
	#if (sys.version_info[0] < 3):
		#raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")
	env = UnityEnvironment(file_name=options.env_name, worker_id = options.workerid, seed =1)
	return env
	
def to_one_hot(ID, N):
	arr = [0]*N
	if ID != -1:
		arr[ID] = 1
	return arr  
																							
def TransObsOnehot(possibleneighbourlist, N):
	arr = [0]*N
	for i in range(0, len(possibleneighbourlist), 2):
		ID = int(possibleneighbourlist[i])
		num_agents = int(possibleneighbourlist[i+1])
		arr[ID] = num_agents
	return arr  

def potential(length, currentZ, desZ, time, goal):
	if desZ == -1:
		return -length[str(currentZ)][str(goal)]    
	else:
		return -length[str(desZ)][str(goal)] - time    

#Change #3:
def randomFunction(env, options, all_static_info, sess):

	edges = all_static_info["edges"]
	#g_1 = all_static_info["g_1"]
	polNNs = all_static_info["polNNs"]
	length = all_static_info["length"]
	T_min = all_static_info["T_min"]
	T_max = all_static_info["T_max"]
	train_mode = all_static_info["train_mode"]
	num_of_agents = all_static_info["num_of_agents"]
	num_of_zones = all_static_info["num_of_zones"]
	GOALS = all_static_info["GOALS"]
	uniqueGOALS = all_static_info["uniqueGOALS"]
	num_of_GOALS = all_static_info["num_of_GOALS"]
	GOALS_index = all_static_info["GOALS_index"]
	cap = all_static_info["cap"] 

	# Things to return in function
	# paths = []
	# Collision = []                
	obsdict = {} 
	actdict = {}
	timeTodestdict = {}
	retdict = {}
	nextretdict = {}

	# Set the default brain to work with
	default_brain = env.brain_names[0]
	brain = env.brains[default_brain]

	# Each trajectory will have at most 100 time steps
	T = options.T

	# Set the discount factor for the problem
	discount = options.discount

	#new episode
	flags = [False] * num_of_agents
	#observations
	states = [[] for _ in range(num_of_agents)]

	states_com2 = [[] for _ in range(num_of_agents)]
	states_com3 = [[] for _ in range(num_of_agents)]
	states_com4 = [[] for _ in range(num_of_agents)]
	observations = [[] for _ in range(num_of_agents)]
	# store sampled actions i
	actions = [[] for _ in range(num_of_agents)]
	actionZones = [[] for _ in range(num_of_agents)]
	actionTimes = [[] for _ in range(num_of_agents)]
	nusParas = [[] for _ in range(num_of_agents)]
	entropys = [[] for _ in range(num_of_agents)]

	# Empirical return for each sampled state-action pair
	rewards = [[] for _ in range(num_of_agents)]
	num_collision = 0

	#dummy actions
	env_info = env.reset(train_mode=train_mode, config={"ReadConfig#0F#1T" : 1.0, "InstanceID" : options.instanceID})[default_brain] 
	action_matrix = np.array([[-1,-1] for _ in range(num_of_agents)])             
	env_info = env.step([action_matrix])[default_brain]
	
	for steps in range(0, T):   
		
		action_matrix = [] 
		actionIDs = [] 
		actionZone = []
		zoneIDs = []
		actionTime = []
		obs = []
		goals = []
		destIDs = []
		timeDests = []
		firstVist = []
		nusPara = []
		entropy = []

		for nn in range(0, num_of_agents): 

			all_obs = env_info.vector_observations[nn]
			agentID = int(all_obs[0])
			zoneID = int(all_obs[1])
			destID = int(all_obs[2])
			timeDest = int(all_obs[3])
			goal = int(all_obs[4])
			
			zoneIDs.append(zoneID)
			goals.append(goal)
			destIDs.append(destID)
			timeDests.append(timeDest)

			possibleneighbourlist = []
			for element in range(5, len(all_obs), 2):
				if all_obs[element] != -1:
					possibleneighbourlist.extend(all_obs[element:element+2])                                                                                                     
			obsOnehot = TransObsOnehot(possibleneighbourlist, num_of_zones) + to_one_hot(agentID, num_of_agents)
			

			firstVist.append(goal)                       

									
			nn_input = np.asarray(obsOnehot).reshape((1, num_of_zones+num_of_agents))
			nn_input1 = np.asarray([0]*num_of_GOALS*len(edges[zoneID])).reshape((1, num_of_GOALS, len(edges[zoneID])))
			obs.append(obsOnehot)
			
			
			if(destID == -1 and timeDest == -1 and (goal != -1 and goal != zoneID)):   
				if len(edges[zoneID]) == 1:
					entropy.append(0)
					a = 0  					
				else:             
					nn_outputs = polNNs[zoneID].getAction(sess, nn_input, nn_input1)                          
					probs = nn_outputs[GOALS_index[goal]][0]   				
					entropy.append(-sum(np.log(probs)*probs)*0.01)				
					a = np.random.choice(len(edges[zoneID]), p=probs)
			
				actOneHot = to_one_hot(a, len(edges[zoneID]))
				nn_input1[0,GOALS_index[goal],:] = actOneHot
				nn_outputs = polNNs[zoneID].getAction(sess, nn_input, nn_input1)   
				nu = nn_outputs[GOALS_index[goal]+num_of_GOALS][0][0]                       
				
				nusPara.append(nu)
				m = T_max - T_min
				timeTodest = T_min + np.random.binomial(m, nu)
				actionIDs.append(a)
				actionZone.append(edges[zoneID][a])
				actionTime.append(timeTodest-T_min)
				action_matrix.append([edges[zoneID][a], timeTodest])
			else:
				actionIDs.append(-1)
				actionZone.append(-1)
				actionTime.append(-1)
				action_matrix.append([-1.0, -1.0])
				nusPara.append(-1)
				entropy.append(-1)

		action_matrix = np.array(action_matrix)      
		env_info = env.step(action_matrix)[default_brain]

		#collect states and actions   
		for nn in range(num_of_agents):
			states[nn].append(zoneIDs[nn])
			states_com2[nn].append(destIDs[nn])
			states_com3[nn].append(timeDests[nn])
			states_com4[nn].append(firstVist[nn])			
			observations[nn].append(obs[nn])
			actions[nn].append(actionIDs[nn])
			actionZones[nn].append(actionZone[nn])
			actionTimes[nn].append(actionTime[nn])
			nusParas[nn].append(nusPara[nn])
			entropys[nn].append(entropy[nn])
	   
		#collect rewards
		#penalty for violating collison
		penaltyAgents = {}
		for group in groupby(sorted(enumerate(zoneIDs), key=lambda x: x[1]), lambda x: x[1]):
			zone = group[0]
			zone_agents = [x[0] for x in group[1]]

			if len(zone_agents) > cap[zone]:
				num_collision += 1
				penalty = -options.penalty
			else:
				penalty = 0		   
			for agent in zone_agents:
				penaltyAgents[agent] = penalty

		for nn in range(0, num_of_agents): 
			if env_info.local_done[nn] == True:
				timeCost = options.final
				flags[nn] = True
			else:                           
				if actionZone[nn] == -1 and flags[nn] == False:
					timeCost = -options.cost
				else:
					timeCost = env_info.rewards[nn]
			 
			if flags[nn] == False:	   
				next_state_single = env_info.vector_observations[nn][1:5]
				phi_next_state = potential(length, int(next_state_single[0]), int(next_state_single[1]), int(next_state_single[2]), int(next_state_single[3]))  
				phi_state = potential(length, zoneIDs[nn], destIDs[nn], timeDests[nn], goals[nn])       
				shaping = discount * phi_next_state - phi_state                                 
			else:
				shaping = 0                       			
			rewards[nn].append(timeCost + shaping + penaltyAgents[nn])
			
		if all(flags):                     
			break  
	
	total_len = 0
	for nn in range(num_of_agents):
		path = states[nn]
		goal_agent = GOALS[nn]
		if path[-1] == goal_agent:
			path = [z for x, y in groupby(path) for z in y if z!=goal_agent] + [goal_agent]
		else:
			path = [z for x, y in groupby(path) for z in y if z!=goal_agent] 		   
		total_len += len(path) -1               


	with open(options.experimentname + '.txt', 'a+') as f:
		f.write(json.dumps(states))
		f.write('\n')

	
	#Calculate empirical return 
	rets = []
	for nn in range(num_of_agents):
		rewards_nn = rewards[nn]
		rets_nn = []
		return_so_far = 0
		for t in range(len(rewards_nn) - 1, -1, -1):
			return_so_far = rewards_nn[t] + discount * return_so_far
			rets_nn.append(return_so_far)
		# The returns are stored backwards in time, so we need to revert it
		rets_nn = np.array(rets_nn[::-1])
		# normalise returns
		rets_nn = (rets_nn - np.mean(rets_nn)) / (np.std(rets_nn) + 1e-8)
		rets.append(rets_nn)
				 
	# # episode ends, samples for training actor network
	for nn in range(num_of_agents):
		
		oneAgent_states = states[nn]
		oneAgent_states4 = states_com4[nn]
		oneAgent_Times = actionTimes[nn]
		oneAgent_observations = observations[nn]
		oneAgent_actions = actions[nn]    

		for ss in range(0, len(oneAgent_states)-1):
			state = oneAgent_states[ss]                                   
			goal = oneAgent_states4[ss]
			tToDes = oneAgent_Times[ss]
			act = oneAgent_actions[ss]
		
			if act != -1:                           
				if state in obsdict:
					obsdict[state].append(oneAgent_observations[ss])
					timeTodestdict[state].append(tToDes)
					act_input = [0]*num_of_GOALS*len(edges[state])
					actOneHot = to_one_hot(act, len(edges[state]))
					act_input[GOALS_index[goal]*len(edges[state]): (GOALS_index[goal]+1)*len(edges[state])] = actOneHot
					actdict[state].append(act_input)					
					retdict[state].append(rets[nn][ss]+entropys[nn][ss])                             
					nextretdict[state].append(rets[nn][ss+1])
					  
				else:
					obsdict[state] = []
					actdict[state] = []
					retdict[state] = []
					timeTodestdict[state] = []
					nextretdict[state] = []                               
					obsdict[state].append(oneAgent_observations[ss])
					timeTodestdict[state].append(tToDes)
					
					act_input = [0]*num_of_GOALS*len(edges[state])
					actOneHot = to_one_hot(act, len(edges[state]))
					act_input[GOALS_index[goal]*len(edges[state]): (GOALS_index[goal]+1)*len(edges[state])] = actOneHot
					actdict[state].append(act_input)					
					retdict[state].append(rets[nn][ss]+entropys[nn][ss])                              
					nextretdict[state].append(rets[nn][ss+1])
	return (total_len, num_collision, obsdict, actdict, timeTodestdict, retdict, nextretdict)
																						   
def trainModel(options, cap, edges, g_1, polNNs, length, listofenvironments, T_min, T_max, train_mode, num_of_agents, num_of_zones, GOALS): 
	


	uniqueGOALS = list(np.unique(GOALS))
	num_of_GOALS = len(uniqueGOALS)
	GOALS_index = {yy : xx for xx, yy in enumerate(uniqueGOALS)}

	with tf.Session(graph=g_1, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('/home/jiajing/Project/REINFORCE/PythonScripts/tflogs/'+options.experimentname, sess.graph)
		
		#We will collect 32 trajectories/episodes per iteration
		N = options.N
		# Each trajectory will have at most 100 time steps
		T = options.T
		# Number of iterations
		n_itr = options.n_itr
		# Set the discount factor for the problem
		discount = options.discount
		#check whethe episode ends
		paths = []
		Collision = []                                            

		# If you need to write the model, just write once here, no need to do it in parallel multiple times.
		WriteModel = False
		if WriteModel:
			# Set the default brain to work with
			env = listofenvironments[0]
			default_brain = env.brain_names[0]
			brain = env.brains[default_brain]
			env_info = env.reset(train_mode=train_mode, config={"WriteModel#0F#1T" : 1.0})[default_brain]
			env.close()
			raise Exception('Please set WriteModel to False now and run again.')    

		pool = Pool(processes=len(listofenvironments)+4)

		all_static_info = {}
		all_static_info["edges"] = edges
		all_static_info["polNNs"] = polNNs
		all_static_info["length"] = length
		all_static_info["T_min"] = T_min
		all_static_info["T_max"] = T_max
		all_static_info["train_mode"] = train_mode
		all_static_info["num_of_agents"] = num_of_agents
		all_static_info["num_of_zones"] = num_of_zones
		all_static_info["GOALS"] = GOALS
		all_static_info["uniqueGOALS"] = uniqueGOALS
		all_static_info["num_of_GOALS"] = num_of_GOALS
		all_static_info["GOALS_index"] = GOALS_index
		all_static_info["cap"] = cap

		for i in range(0,n_itr): 
			#new iteration
			with open(options.experimentname + '.txt', 'a+') as f:
				f.write('Iteration'+str(i)+'\n')
			
			obsdict = {} 
			actdict = {}
			timeTodestdict = {}
			retdict = {}
			nextretdict = {}
			
			result = pool.amap(randomFunction, listofenvironments, [options]*len(listofenvironments), [all_static_info]*len(listofenvironments), [sess]*len(listofenvironments))
			all_dictionaries_all_episodes = result.get()

			for data_from_env in range(0,N):  
				total_len_eps, num_collision_eps, obsdict_eps, actdict_eps, timeTodestdict_eps, retdict_eps, nextretdict_eps = all_dictionaries_all_episodes[data_from_env]				
				for x in range(0, num_of_zones):
					if x in obsdict_eps:
						if x in obsdict:
							obsdict[x].extend(obsdict_eps[x])
							actdict[x].extend(actdict_eps[x])
							timeTodestdict[x].extend(timeTodestdict_eps[x])
							retdict[x].extend(retdict_eps[x])
							nextretdict[x].extend(nextretdict_eps[x])
						else:
							obsdict[x] = []
							actdict[x] = []
							timeTodestdict[x] = []
							retdict[x] = []
							nextretdict[x] = []
							obsdict[x].extend(obsdict_eps[x])
							actdict[x].extend(actdict_eps[x])
							timeTodestdict[x].extend(timeTodestdict_eps[x])
							retdict[x].extend(retdict_eps[x])
							nextretdict[x].extend(nextretdict_eps[x])
				paths.append(total_len_eps)
				Collision.append(num_collision_eps)   
					  
			#iteration ends, start training Actor network
			loss2_before = 0
			loss2_after = 0
			for x in range(0, num_of_zones):
				inpdict = {}
				if x in obsdict:
					inpdict[polNNs[x].input_var] = np.asarray(obsdict[x]).reshape((len(obsdict[x]), num_of_zones+num_of_agents))

					inpdict[polNNs[x].input_var1] = np.asarray(actdict[x]).reshape((len(actdict[x]), num_of_GOALS, len(edges[x])))
					
					inpdict[polNNs[x].act_var] = np.asarray(actdict[x]).reshape((len(actdict[x]), num_of_GOALS, len(edges[x])))

					inpdict[polNNs[x].ret_var] = np.asarray(retdict[x]).reshape((1, len(retdict[x])))
					
					inpdict[polNNs[x].nextret_var] = np.asarray(nextretdict[x]).reshape((1,len(nextretdict[x])))

					inpdict[polNNs[x].phi] = np.asarray(timeTodestdict[x]).reshape((1,len(timeTodestdict[x])))  

					#learning rate decaying
					# polNNs[x].learning_rate =  options.lr * 0.96**(i/5)       
					loss2_before += sess.run(polNNs[x].loss, feed_dict=inpdict)
					sess.run(polNNs[x].learning_step, feed_dict=inpdict)
					loss2_after += sess.run(polNNs[x].loss, feed_dict=inpdict)

			summary = tf.Summary()            
			summary.value.add(tag='summaries/length_of_path', simple_value = np.mean(paths[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/num_collision', simple_value = np.mean(Collision[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/actor_training_loss', simple_value = (loss2_before+loss2_after)/2)
					  
			writer.add_summary(summary, i)          
			writer.flush()     
		saver = tf.train.Saver()
		save_path = '/home/jiajing/Project/REINFORCE/PythonScripts/trainedModel/'+options.experimentname
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		saver.save(sess, save_path+'/model.ckpt') 
		for env in listofenvironments:				
			env.close()
	

def main():

	# Whether to run the environment in training or inference mode
	train_mode = False 
	parser = OptionParser()
	parser.add_option("-t", "--time", type='int', dest='T', default=500)
	parser.add_option("-b", "--batch", type='int', dest='N', default=8)
	parser.add_option("-i", "--iteration", type='int', dest='n_itr', default=500)    
	parser.add_option("-d", "--discount", type='float', dest='discount', default=0.99)    
	parser.add_option("-e", "--envname", type='str', dest='env_name', default='/home/jiajing/Project/REINFORCE/Env/10x10')    
	parser.add_option("-m", "--modelpath", type='str', dest='modelPath', default='/home/jiajing/Project/REINFORCE/Models/10x10.model')    
	parser.add_option("-x", "--experimentname", type='str', dest='experimentname', default='ex0')   
	parser.add_option("-s", "--instanceID", type='float', dest='instanceID', default=0) 
	parser.add_option("-w", "--workerid", type='int', dest='workerid', default=0)
	parser.add_option("-l", "--learningrate", type='float', dest='lr', default=0.001)
	parser.add_option("-c", "--nonactcost", type='float', dest='cost', default=1)
	parser.add_option("-f", "--finalcost", type='float', dest='final', default=100)
	parser.add_option("-p", "--penalty", type='float', dest='penalty', default=50)
	(options, args) = parser.parse_args()  

	configPath = '/home/jiajing/Project/REINFORCE/ExtraConfigs/10x10_ex'+str(int(options.instanceID))+'.txt'
	f = open(configPath, 'r')
	configfile = f.read().split('\n')
	T_min = int(configfile[1])
	T_max = int(configfile[3])
	max_cap = int(configfile[5])
	num_of_zones = int(configfile[7])
	num_of_agents = int(configfile[9])
	start_zones = json.loads(configfile[11])
	goal_zones = json.loads(configfile[13])
	GOALS = json.loads(configfile[17])
	start_goal = start_zones + goal_zones
	max_capacity = max_cap
	cap = {}
	np.random.seed(666)
	for nn in range(num_of_zones):  
		if nn in start_goal:
			cap[nn] = num_of_agents
		else:
			cap[nn] = np.random.randint(1,1+max_capacity)

	length = shortestPath(options.modelPath)
	edges = ConstructGraph(options.modelPath)
	g_1 = tf.Graph()      
	polNNs = InitialiseActor(g_1, options.lr, edges, num_of_zones, num_of_agents, GOALS, T_min, T_max)

	listofenvironments = []
	worker = options.workerid
	for i in range(0, options.N):
		options.workerid = i + worker
		env = loadUnityEnvironment(options)
		listofenvironments.append(env)

	trainModel(options, cap, edges, g_1, polNNs, length, listofenvironments, T_min, T_max, train_mode, num_of_agents, num_of_zones, GOALS)

if __name__ == '__main__':
	main()























