# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:32:08 2019

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
from pathos.multiprocessing import ThreadingPool as Pool
import os 

h1 = 64
h2 = 64
h3 = 64
h4 = 64
h5 = 64

class Actor:
	def __init__(self, T_max, num_of_agents, num_of_zones, s, obs, dis_actions):
		self.num_of_agents = num_of_agents
		self.num_of_zones = num_of_zones
		self.len_state_single = num_of_zones*2+T_max+1
		self.obs = obs
		self.global_state = s
		self.dis_actions = dis_actions
			  
		
		self.learning_rate = None
		self.optimizer = None
		self.learning_step_single = None
		self.learning_step = {}

		with tf.variable_scope('Mu'):
			self.output_var = self.generate_mu(scope='eval', trainable=True)
			self.output_var_target = self.generate_mu(scope='target', trainable=False)

		self.model_weights = {}
		for nn in range(self.num_of_agents): 
			self.model_weights[nn] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Mu/eval/agent'+str(nn))
		
	def setCompGraph(self):
		with tf.variable_scope('optimisation'):
			self.action_gradients = tf.placeholder(shape=[None, self.num_of_agents, 1],dtype=tf.float32)
			for nn in range(self.num_of_agents):
				action_gradients_single = tf.slice(self.action_gradients, [0, nn, 0], [-1, 1, 1])
				
				action_gradients_single = tf.reshape(tensor=action_gradients_single, shape=(-1, 1))
				
				self.parameter_gradietns_single = tf.gradients(self.output_var[nn], self.model_weights[nn], -action_gradients_single)

				self.gradients_single = zip(self.parameter_gradietns_single, self.model_weights[nn])

				# Learning Rate
				self.learning_rate = 0.001

				# Defining Optimizer
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

				# Update Gradients
				self.learning_step_single = self.optimizer.apply_gradients(self.gradients_single)

				self.learning_step[nn] = self.learning_step_single

	def generate_mu(self, scope, trainable):
		with tf.variable_scope(scope):
			con_action_output = {}
			for nn in range(self.num_of_agents):    

				with tf.variable_scope('agent'+str(nn)):

					state_single = tf.slice(self.global_state, [0, nn, 0], [-1, 1, self.len_state_single])
					state_single = tf.reshape(tensor=state_single, shape=(-1, self.len_state_single))

					obs_single = tf.slice(self.obs, [0, nn, 0], [-1, 1, self.num_of_zones])
					obs_single = tf.reshape(tensor=obs_single, shape=(-1, self.num_of_zones))
					
					dis_action_single = tf.slice(self.dis_actions, [0, nn, 0], [-1, 1, 4])
					dis_action_single = tf.reshape(tensor=dis_action_single, shape=(-1, 4))
	
	
					# generate mu network
					mu_hidden_1 = tf.layers.dense(inputs=tf.concat([state_single, obs_single, dis_action_single], 1), units=h1, activation=tf.nn.relu,
											# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
											# bias_initializer=tf.constant_initializer(0.00),  # biases
											use_bias=True,
											trainable=trainable, name='mu_dense_h1_agent'+str(nn))
					mu_hidden_2 = tf.layers.dense(inputs=mu_hidden_1, units=h2, 
											# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
											# bias_initializer=tf.constant_initializer(0.00),  # biases
											use_bias=True,
											trainable=trainable, name='mu_dense_h2_agent'+str(nn))
					con_action_single = tf.layers.dense(inputs=mu_hidden_2, units=1, activation=tf.nn.sigmoid,
											# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
											# bias_initializer=tf.constant_initializer(0.00),  # biases
											use_bias=True,
											trainable=trainable, name='mu_dense_h3_agent'+str(nn))
	
					con_action_output[nn] = con_action_single
			
			return con_action_output

	def getAction(self, sess, global_state, obs, dis_actions):
		return sess.run(self.output_var, feed_dict={self.global_state : global_state, self.obs : obs, self.dis_actions : dis_actions})

	def getAction_target(self, sess, global_state, obs, dis_actions):
		return sess.run(self.output_var_target, feed_dict={self.global_state : global_state, self.obs : obs, self.dis_actions : dis_actions})

class Critic:
	def __init__(self, T_max, num_of_agents, num_of_zones, s, obs, dis_actions, con_actions):
		self.num_of_agents = num_of_agents
		self.num_of_zones = num_of_zones
		self.len_state_single = num_of_zones*2+T_max+1
		self.global_state = s
		self.obs = obs
		self.dis_actions = dis_actions
		self.con_actions = con_actions
		self.gamma = 0.99      
		self.q_value_mix_next = None
		self.r = None         
		self.td_error = None
		self.loss = None
		self.learning_rate = None
		self.optimizer = None
		self.learning_step = None

		with tf.variable_scope('qmix'):   
			self.output_var = self.generate_qmix(scope='eval', trainable=True)
			self.output_var_target = self.generate_qmix(scope='target', trainable=False)
		self.action_gradients = tf.gradients(self.output_var[1], self.con_actions)

	def setCompGraph(self):
		with tf.variable_scope('optimisation'):   
			self.q_value_mix_next = tf.placeholder(shape=[None, 1], dtype=tf.float32)
			self.r = tf.placeholder(shape=[None, 1], dtype=tf.float32)					
			self.td_error = self.r + self.gamma * self.q_value_mix_next - self.output_var[1]
			self.loss = tf.reduce_mean(tf.square(self.td_error))
			
			# Learning Rate
			self.learning_rate = 0.01

			# Defining Optimizer
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

			# Update Gradients
			self.learning_step = (self.optimizer.minimize(self.loss))

	
	def generate_qmix(self, scope, trainable):
		
		with tf.variable_scope(scope):
			
			q_values_output = {}

			for nn in range(self.num_of_agents):
				
				state_single = tf.slice(self.global_state, [0, nn, 0], [-1, 1, self.len_state_single])
				state_single = tf.reshape(tensor=state_single, shape=(-1, self.len_state_single))

				obs_single = tf.slice(self.obs, [0, nn, 0], [-1, 1, self.num_of_zones])
				obs_single = tf.reshape(tensor=obs_single, shape=(-1, self.num_of_zones))

				dis_action_single = tf.slice(self.dis_actions, [0, nn, 0], [-1, 1, 4])
				dis_action_single = tf.reshape(tensor=dis_action_single, shape=(-1, 4))
				
				con_action_single = tf.slice(self.con_actions, [0, nn, 0], [-1, 1, 1])
				con_action_single = tf.reshape(tensor=con_action_single, shape=(-1, 1))

				# generate_single_q_network
				q_hidden_1 = tf.layers.dense(inputs=tf.concat([state_single, obs_single, dis_action_single, con_action_single], 1), units=h1, activation=tf.nn.relu,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='q_dense_h1_agent'+str(nn))
				q_hidden_2 = tf.layers.dense(inputs=q_hidden_1, units=h2, activation=tf.nn.relu,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='q_dense_h2_agent'+str(nn))
				q_hidden_3 = tf.layers.dense(inputs=q_hidden_2, units=h3, activation=tf.nn.relu,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='q_dense_h3_agent'+str(nn))
				q_value = tf.layers.dense(inputs=q_hidden_3, units=1,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='dense_h4_agent'+str(nn))
				q_values_output[nn] = q_value
		
			q_value_list = q_values_output.values()
			q_values = tf.concat([tensor for tensor in q_value_list], axis=1)

			#tge hypernetworks take the global state s as the input and outputs the weights of the feedforward network
			s = tf.reshape(self.global_state, [-1, self.len_state_single*self.num_of_agents])
			w1 = tf.layers.dense(s, self.num_of_agents * h3,
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_w1')
			w2 = tf.layers.dense(s, h3 * 1, 
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_w2')
			b1 = tf.layers.dense(s, h3,
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_b1')
			b2_h = tf.layers.dense(s, h3,  activation=tf.nn.relu,
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_b2_h')
			b2 = tf.layers.dense(b2_h, 1, 
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_b2')
			w1 = tf.abs(w1)
			w1 = tf.reshape(w1, [-1, self.num_of_agents, h3])
			w2 = tf.abs(w2)
			w2 = tf.reshape(w2, [-1, h3, 1])
			q_values = tf.reshape(q_values, [-1,1,self.num_of_agents])
			q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1),[-1,h3]) )  + b1
			q_hidden = tf.reshape(q_hidden, [-1,1,h3])
			q_value_mix = tf.reshape(tf.matmul(q_hidden, w2),[-1,1]) + b2

			return q_values_output, q_value_mix

	def get_q_values(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.output_var, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})[0]

	def get_q_values_target(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.output_var_target, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})[0]

	def get_q_value_mix_target(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.output_var_target, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})[1]

	def get_gradients(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.action_gradients, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})

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
	# num_of_zones = int(model[0].strip().split(':')[-1].strip())
	edges = {}
	for i in range(3, len(model)):
		if(len(model[i]) == 0):
			continue
		edgeRecords = model[i].strip().split(' ')
#         edges[int(edgeRecords[0])] = [int(edgeRecords[0])] + [int(x) for x in edgeRecords[1:]]
		edges[int(edgeRecords[0])] = [int(x) for x in edgeRecords[1:]]      
	return edges

def InitialiseActor(g_1, T_max, num_of_agents, num_of_zones):
	with g_1.as_default():
		global_s = tf.placeholder(shape=[None, num_of_agents, num_of_zones*2+T_max+1], dtype=tf.float32)
		obs = tf.placeholder(shape=[None, num_of_agents, num_of_zones], dtype=tf.float32)
		dis_actions = tf.placeholder(shape=[None, num_of_agents, 4], dtype=tf.float32)
		ConNN = Actor(T_max=T_max, num_of_agents=num_of_agents, num_of_zones=num_of_zones, s=global_s, obs=obs, dis_actions=dis_actions)
		ConNN.setCompGraph()

	return ConNN

def InitialiseCritic(g_1, T_max, num_of_agents, num_of_zones):
	with g_1.as_default():
		global_s = tf.placeholder(shape=[None, num_of_agents, num_of_zones*2+T_max+1], dtype=tf.float32)
		obs = tf.placeholder(shape=[None, num_of_agents, num_of_zones], dtype=tf.float32)
		con_actions = tf.placeholder(shape=[None, num_of_agents, 1], dtype=tf.float32)
		dis_actions = tf.placeholder(shape=[None, num_of_agents, 4], dtype=tf.float32)
		DisNN = Critic(T_max=T_max, num_of_agents=num_of_agents, num_of_zones=num_of_zones, s=global_s, obs=obs, dis_actions=dis_actions, con_actions=con_actions)
		DisNN.setCompGraph()

	return DisNN

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
																							   
def randomFunction(env, eps_threshold, options, all_static_info, sess):

	edges = all_static_info["edges"]
	#g_1 = all_static_info["g_1"]
	ConNN = all_static_info["ConNN"]
	DisNN = all_static_info["DisNN"] 
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
	observations = []
	global_states = []
	# store sampled actions i
	actions = []
	actionZones = [[] for _ in range(num_of_agents)]
	actionTimes = [[] for _ in range(num_of_agents)]
	nusParas = []
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
		nusPara = []
		entropy = []
		global_state = []
		hybrid_actions = {}
		
		for nn in range(0, num_of_agents): 

			obs_single = env_info.vector_observations[nn]
			agentID = int(obs_single[0])
			zoneID = int(obs_single[1])
			destID = int(obs_single[2])
			timeDest = int(obs_single[3])
			goal = int(obs_single[4])
			firstVisit = 1 if goal != -1 else 0
			zoneIDs.append(zoneID)
			goals.append(goal)
			destIDs.append(destID)
			timeDests.append(timeDest)
	
			possibleneighbourlist = []
			for element in range(5, len(obs_single), 2):
				if obs_single[element] != -1:
					possibleneighbourlist.extend(obs_single[element:element+2])                                                                                                     
			obsOnehot = TransObsOnehot(possibleneighbourlist, num_of_zones)
			obs.append(obsOnehot)
			global_state.append(to_one_hot(zoneID, num_of_zones)+to_one_hot(destID, num_of_zones)+to_one_hot(timeDest, T_max)+[firstVisit])
			
			hybrid_actions[nn] = [[],[]]


		obs_input = np.asarray(obs).reshape((1, num_of_agents, num_of_zones))
		global_state_input = np.asarray(global_state).reshape((1, num_of_agents, num_of_zones*2+T_max+1))
		
		for all_act in range(4):	
			dis_actions_input = np.asarray([to_one_hot(all_act, 4)] * num_of_agents).reshape((1, num_of_agents, 4))						
			
			con_actions = ConNN.getAction(sess, global_state_input, obs_input, dis_actions_input)        
			
			con_actions_input = np.asarray([xx[0][0] for xx in con_actions.values()]).reshape((1, num_of_agents, 1))	
			# con_actions_input = np.asarray([0 for xx in con_actions.values()]).reshape((1, num_of_agents, 1))	
			q_values = DisNN.get_q_values(sess, global_state_input, obs_input, dis_actions_input, con_actions_input)

			for nn in range(0, num_of_agents): 
				hybrid_actions[nn][0].append(con_actions[nn][0][0])
				hybrid_actions[nn][1].append(q_values[nn][0][0])


		for nn in range(0, num_of_agents): 	
			if(destIDs[nn] == -1 and timeDests[nn] == -1 and (goals[nn] != -1 and goals[nn] != zoneIDs[nn])):   			
				possible_actions = [p_a for p_a in range(len(edges[zoneIDs[nn]]))]
				if np.random.rand() < eps_threshold:
					a = np.random.choice(possible_actions)
				#max q_value
				else:
					a = possible_actions[np.argmax([hybrid_actions[nn][1][p_a] for p_a in possible_actions])]
				   
				nu = hybrid_actions[nn][0][a] + 0.1*np.random.randn(1)[0]
				nu = np.clip(nu, 0, 1)
				nusPara.append(nu)
				m = T_max - T_min
				timeTodest = T_min + np.random.binomial(m, nu)
				actionIDs.append(to_one_hot(a, 4))
				actionZone.append(edges[zoneIDs[nn]][a])
				actionTime.append(timeTodest-T_min)
				action_matrix.append([edges[zoneIDs[nn]][a], timeTodest])
			else:
				actionIDs.append(to_one_hot(-1, 4))
				actionZone.append(-1)
				actionTime.append(-1)
				action_matrix.append([-1.0, -1.0])
				nusPara.append(-1)


		action_matrix = np.array(action_matrix)      
		env_info = env.step(action_matrix)[default_brain]

		# #collect states and actions
		observations.append(obs)  
		global_states.append(global_state)
		nusParas.append(nusPara)
		actions.append(actionIDs)
		
		for nn in range(num_of_agents):
			states[nn].append(zoneIDs[nn])
			actionZones[nn].append(actionZone[nn])

		# #collect rewards
		# #penalty for violating collison
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
				# timeCost = 0
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
			
			rewards[nn].append(timeCost + penaltyAgents[nn])

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


	sample_len = len(rewards[0])
	global_rewards = np.asarray(rewards).reshape((num_of_agents,sample_len))
	global_rewards = list(np.sum(global_rewards,axis=0)[1:])
	#episode ends, samples for training critic network
	samples_global_states = np.asarray(global_states).reshape((sample_len, num_of_agents, num_of_zones*2+T_max+1))
	samples_observations = np.asarray(observations).reshape((sample_len, num_of_agents, num_of_zones))
	samples_dis_actions = np.asarray(actions).reshape((sample_len, num_of_agents, 4))
	samples_con_actions = np.asarray(nusParas).reshape((sample_len, num_of_agents, 1))
	samples_states = np.asarray(states).reshape((num_of_agents,sample_len)).T
	
	return (all(flags), total_len, num_collision, samples_states, samples_global_states, samples_observations, samples_dis_actions, samples_con_actions, global_rewards)
																			   
def trainModel(options, cap, edges, g_1, ConNN, DisNN, length, listofenvironments, T_min, T_max, train_mode, num_of_agents, num_of_zones, GOALS): 
	


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
		all_static_info["ConNN"] = ConNN
		all_static_info["DisNN"] = DisNN
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
			
			QMIXParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qmix/eval')
			QMIXTargetParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qmix/target') 
			MuParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Mu/eval')
			MuTargetParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Mu/target') 
			soft_replacement = [tf.assign(t, 0.99 * t + 0.01* e) for t, e in zip(QMIXTargetParameters,QMIXParameters)]
			sess.run(soft_replacement)
			soft_replacement = [tf.assign(t, 0.99 * t + 0.01* e) for t, e in zip(MuTargetParameters,MuParameters)]
			sess.run(soft_replacement) 

			#new iteration
			with open(options.experimentname + '.txt', 'a+') as f:
				f.write('Iteration'+str(i)+'\n')
			
			#did not consider the case that last state is terminal state.
			all_samples_global_states = []
			all_samples_next_global_states = []
			all_samples_last_global_states = []
			all_samples_observations = []
			all_samples_next_observations = []
			all_samples_last_observations = []
			all_samples_dis_actions = []
			all_samples_next_dis_actions = []
			all_samples_last_dis_actions = []
			all_samples_con_actions = []
			all_samples_next_con_actions = []
			all_samples_last_con_actions = []
			all_rets = []
			all_last_rets = []
			all_samples_next_states = [] 

			eps_threshold = 0.05 + (0.9 - 0.05) * np.exp(-1. * i / 300)

			result = pool.amap(randomFunction, listofenvironments, [eps_threshold]*len(listofenvironments), [options]*len(listofenvironments), [all_static_info]*len(listofenvironments), [sess]*len(listofenvironments))
			all_dictionaries_all_episodes = result.get()

			for data_from_env in range(0,N):  

				all_reach, total_len_eps, num_collision_eps, samples_states, samples_global_states, samples_observations, samples_dis_actions, samples_con_actions, rets = all_dictionaries_all_episodes[data_from_env]	
				
				all_samples_global_states.append(samples_global_states[0:-2,:,:])
				all_samples_next_global_states.append(samples_global_states[1:-1,:,:])
				all_samples_observations.append(samples_observations[0:-2,:,:])
				all_samples_next_observations.append(samples_observations[1:-1,:,:])
				all_samples_dis_actions.append(samples_dis_actions[0:-2,:,:])
				all_samples_next_dis_actions.append(samples_dis_actions[1:-1,:,:])
				all_samples_con_actions.append(samples_con_actions[0:-2,:,:])
				all_samples_next_con_actions.append(samples_con_actions[1:-1,:,:])
				all_rets.append(rets[0:-1])
				all_samples_next_states.append(samples_states[1:-1,:])

				if all_reach:
			
					all_samples_last_global_states.append(samples_global_states[-2:-1,:,:])
				  
					all_samples_last_observations.append(samples_observations[-2:-1,:,:])

					all_samples_last_dis_actions.append(samples_dis_actions[-2:-1,:,:])
			 
					all_samples_last_con_actions.append(samples_con_actions[-2:-1,:,:])

					all_last_rets.append([rets[-1]])

				paths.append(total_len_eps)
				Collision.append(num_collision_eps)  
			
			all_samples_global_states = np.concatenate(all_samples_global_states, axis=0)
			all_samples_next_global_states = np.concatenate(all_samples_next_global_states, axis=0)
			
			all_samples_observations = np.concatenate(all_samples_observations, axis=0)
			all_samples_next_observations = np.concatenate(all_samples_next_observations, axis=0)
			
			all_samples_dis_actions = np.concatenate(all_samples_dis_actions, axis=0)
			all_samples_next_dis_actions = np.concatenate(all_samples_next_dis_actions, axis=0)
					
			all_samples_con_actions = np.concatenate(all_samples_con_actions, axis=0)
			all_samples_next_con_actions = np.concatenate(all_samples_next_con_actions, axis=0)
			
			all_rets = np.concatenate(all_rets, axis=0)
			  
			all_samples_next_states = np.concatenate(all_samples_next_states, axis=0)


			if len(all_last_rets) != 0:
				all_samples_last_global_states = np.concatenate(all_samples_last_global_states, axis=0)
				all_samples_last_observations = np.concatenate(all_samples_last_observations, axis=0)
				all_samples_last_dis_actions = np.concatenate(all_samples_last_dis_actions, axis=0)
				all_samples_last_con_actions = np.concatenate(all_samples_last_con_actions, axis=0)
				all_last_rets = np.concatenate(all_last_rets, axis=0)

			hybrid_actions = {}
			for nn in range(0, num_of_agents):  
				hybrid_actions[nn] = [[],[]]
			sample_len = all_samples_observations.shape[0]
			for all_act in range(4):	
				dis_actions_input = np.asarray([to_one_hot(all_act, 4)] * num_of_agents * sample_len).reshape((sample_len, num_of_agents, 4))						
				
				con_actions = ConNN.getAction_target(sess, all_samples_next_global_states, all_samples_next_observations, dis_actions_input)        
				
				con_actions_input = np.asarray([xx[:,0] for xx in con_actions.values()]).T.reshape((sample_len, num_of_agents, 1))

				q_values_target = DisNN.get_q_values_target(sess, all_samples_next_global_states, all_samples_next_observations, dis_actions_input, con_actions_input)

				for nn in range(0, num_of_agents): 
					hybrid_actions[nn][0].append(con_actions[nn][:,0])
					hybrid_actions[nn][1].append(q_values_target[nn][:,0])

			for nn in range(0, num_of_agents): 
				hybrid_actions[nn][0] = np.asarray(hybrid_actions[nn][0]).T
				hybrid_actions[nn][1] = np.asarray(hybrid_actions[nn][1]).T

			dis_actions_input = []
			con_actions_input = []

			for tt in range(sample_len):
				tmp_dic_actions = []
				tmp_con_actions = []
				for nn in range(0, num_of_agents):
					if all_samples_next_con_actions[tt][nn][0] == -1:
						a = -1 
						nu = -1                       
					else:
						a = np.argmax(hybrid_actions[nn][1][tt][0:len(edges[all_samples_next_states[tt][nn]])])	
						nu = hybrid_actions[nn][0][tt][a]
					tmp_dic_actions.append(to_one_hot(a, 4))
					tmp_con_actions.append(nu)
				dis_actions_input.append(tmp_dic_actions)
				con_actions_input.append(tmp_con_actions)

			dis_actions_input = np.asarray(dis_actions_input).reshape((sample_len, num_of_agents, 4))
			con_actions_input = np.asarray(con_actions_input).reshape((sample_len, num_of_agents, 1))
			# con_actions_input = np.asarray([0]*sample_len*num_of_agents).reshape((sample_len, num_of_agents, 1))	

			q_value_mix_next = DisNN.get_q_value_mix_target(sess, all_samples_next_global_states, all_samples_next_observations, dis_actions_input, con_actions_input)
			# q_value_mix_next = DisNN.get_q_value_mix_target(sess, all_samples_next_global_states, all_samples_next_observations, all_samples_next_dis_actions, all_samples_next_con_actions)
			#starting training critic
			inpdict = {}
			if len(all_last_rets) != 0:
				inpdict[DisNN.global_state] = np.concatenate([all_samples_global_states, all_samples_last_global_states], axis=0)
				
				inpdict[DisNN.obs] = np.concatenate([all_samples_observations, all_samples_last_observations], axis=0)
	
				inpdict[DisNN.dis_actions] = np.concatenate([all_samples_dis_actions, all_samples_last_dis_actions], axis=0)
						
				inpdict[DisNN.con_actions] = np.concatenate([all_samples_con_actions, all_samples_last_con_actions], axis=0)
	
				inpdict[DisNN.r] = np.concatenate([all_rets.reshape((len(all_rets),1)), all_last_rets.reshape((len(all_last_rets),1))], axis=0) 
	
				inpdict[DisNN.q_value_mix_next] =  np.concatenate([q_value_mix_next, np.asarray([0]*len(all_last_rets)).reshape((len(all_last_rets),1))], axis=0) 
			
			else:
				inpdict[DisNN.global_state] = all_samples_global_states
				
				inpdict[DisNN.obs] = all_samples_observations
	
				inpdict[DisNN.dis_actions] = all_samples_dis_actions
						
				inpdict[DisNN.con_actions] = all_samples_con_actions
	
				inpdict[DisNN.r] = all_rets.reshape((len(all_rets),1))
	
				inpdict[DisNN.q_value_mix_next] =  q_value_mix_next
			
			
			sess.run(DisNN.learning_step, feed_dict=inpdict)
			loss1_after = sess.run(DisNN.loss, feed_dict=inpdict)
			
			#starting training actor
			gradients = DisNN.get_gradients(sess, all_samples_global_states, all_samples_observations, all_samples_dis_actions, all_samples_con_actions)[0]

			inpdicts = {}
			all_samples_global_states_single = {}
			all_samples_observations_single = {}
			all_samples_dis_actions_single = {}
			gradients_single ={}

			for nn in range(num_of_agents):
				inpdicts[nn] = {}
				all_samples_global_states_single[nn] = []
				all_samples_observations_single[nn] = []
				all_samples_dis_actions_single[nn] = []
				gradients_single[nn] =[]

			sample_len = all_samples_con_actions.shape[0]

			for tt in range(sample_len):
				for nn in range(num_of_agents):
					if all_samples_con_actions[tt][nn][0] != -1:
						all_samples_global_states_single[nn].append(all_samples_global_states[tt,:,:])
						all_samples_observations_single[nn].append(all_samples_observations[tt,:,:])
						all_samples_dis_actions_single[nn].append(all_samples_dis_actions[tt,:,:])
						gradients_single[nn].append(gradients[tt,:,:])
	
	
			for nn in range(num_of_agents):

				inpdicts[nn][ConNN.global_state] = np.asarray(all_samples_global_states_single[nn]).reshape((-1, num_of_agents, num_of_zones*2+T_max+1))
			
				inpdicts[nn][ConNN.obs] = np.asarray(all_samples_observations_single[nn]).reshape((-1, num_of_agents, num_of_zones))

				inpdicts[nn][ConNN.dis_actions] = np.asarray(all_samples_dis_actions_single[nn]).reshape((-1, num_of_agents, 4))
					
				inpdicts[nn][ConNN.action_gradients] = np.asarray(gradients_single[nn]).reshape((-1, num_of_agents, 1))

				sess.run(ConNN.learning_step[nn], feed_dict=inpdicts[nn])

				
			summary = tf.Summary()            
			summary.value.add(tag='summaries/length_of_path', simple_value = np.mean(paths[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/num_collision', simple_value = np.mean(Collision[i*N : (i+1)*N]))
			# summary.value.add(tag='summaries/actor_training_loss', simple_value = (loss2_before+loss2_after)/2)
			summary.value.add(tag='summaries/critic_training_loss', simple_value = loss1_after)		  
			writer.add_summary(summary, i)          
			writer.flush()     
# # #            print(str(loss_before) + " " + str(loss_after) + " ")
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
	parser.add_option("-c", "--nonactcost", type='float', dest='cost', default=2)
	parser.add_option("-f", "--finalcost", type='float', dest='final', default=100)
	parser.add_option("-p", "--penalty", type='float', dest='penalty', default=10)
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

	ConNN = InitialiseActor(g_1, T_max, num_of_agents, num_of_zones)
	DisNN = InitialiseCritic(g_1, T_max, num_of_agents, num_of_zones)

	listofenvironments = []
	worker = options.workerid
	for i in range(0, options.N):
		options.workerid = i + worker
		env = loadUnityEnvironment(options)
		listofenvironments.append(env)

	trainModel(options, cap, edges, g_1, ConNN, DisNN, length, listofenvironments, T_min, T_max, train_mode, num_of_agents, num_of_zones, GOALS)

if __name__ == '__main__':
	main()