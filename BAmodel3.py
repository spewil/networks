from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
import random as rd
from scipy.optimize import curve_fit
import json 
import time
from log_bin import *
from collections import Counter
from numba import jit

# @jit
# @profile
def add_nodes3(nodelist,m,num_added):

	# index list of connected node pairs [node1,node2,node1,node3,...]
	nodes = list(nodelist)

	# i is the node we're adding 
	# between 1 more than the last index of the og graph
	# and the number we want to add 
	for i in range(len(nodelist),int(num_added)):

		rand_nodes = []
		# choose m random nodes 
		while len(rand_nodes) < m:
			# choose a random one 
			# rand_node = rd.choice(nodes)
			# random integer of indices of nodes -- [0,length-1] 
			rand_node = nodes[rd.randint(0,len(nodes)-1)]

			# if chosen value is already in list 
			if rand_node in rand_nodes:
				pass	
			# otherwise add it to random node list 
			else: 
				rand_nodes.append(rand_node)
				# add the recent node and its connection 
				nodes.extend([i,rand_node])

	return nodes  

def run_BA3(m,num_added,num_trials,save): 
	# generate a starting graph 
	# 50 nodes to start 
	# complete graph to start, p_edge = .3
	# want at least the number to connect to 
	G=nx.complete_graph(2*m+1)

	# the starting graph list 
	nodes = [y for x in G.edges() for y in x]

	# run BA algo 
	degree_list = []
	for i in range(num_trials):

		# tic = time.clock()
		# jit this to make faster 
		nodes = add_nodes3(nodes,m,num_added)
		
		# make list of node degree counts 
		degrees = Counter(nodes).values()

		# toc = time.clock()
		# print the runtime 
		# print 'trial runtime is ' + str(toc - tic)

		# add new data 
		degree_list.extend(degrees)

	# save degree distro data to json 
	if save == True:
		file_path = '../DATA/degrees_' + str(int(np.log10(num_added))) + '_' + str(m) + '_' + str(num_trials) + '.json'
		with open(file_path, 'w') as fp:
				json.dump(degree_list, fp) 

	return degree_list