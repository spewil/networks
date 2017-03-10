from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from scipy.optimize import curve_fit
import json 
import time
from log_bin import *
from collections import Counter
from numba import jit

@jit#(nopython=True)
# @profile
def add_nodes(nodelist,edgelist,m,num_added):
	
	# start with an array of nodes [d1,d2,d3,...] and edges [(n,m),(x,y),...] 
	N = int(num_added)
	# the keys for the nodes 
	startNo = nodelist.size
	indices = np.arange(startNo + num_added)
	# grab the list of node degrees
	degrees = np.concatenate((nodelist,np.zeros(N,dtype=np.int32)),axis=0)
	# edge list 
	# edges = np.append(np.array(edgelist,np.zeros(N)))

	for i in np.arange(num_added):

		# # empty list of randomly chosen nodes 
		# rand_nodes = np.array([])

		# list of probabilities node_degree/total outgoing edges 
		probs = degrees/np.sum(degrees)

		# make a nonuniform choice of m nodes from the list of nodes
		# before adding a node! to protect against self edges 
		# probability based linearly on degree, P(k) ~ k 
		# returns ndarray([node_indices])
		rand_nodes = np.random.choice(indices, m, replace=False, p=probs)

		# adds a node to the end with degree m  
		degrees[i+startNo] = m

		# update list of node keys 
		indices[i+startNo] = i+startNo

		# add a connection to the randomly chosen nodes  
		degrees[rand_nodes] += 1 

		# add the new edges 
		# edges = np.append(edges,np.array(zip(rand_nodes,np.ones(m)*len(indices)))[:])

	return degrees 

def run_BA(m,num_added,num_trials,save): 
	# generate a starting graph 
	# 50 nodes to start 
	# complete graph to start, p_edge = .3
	# want at least the number to connect to 
	G=nx.complete_graph(2*m+1)
	nodes = np.array(G.degree().values(),dtype=np.int32)
	edges = np.array(G.edges(),dtype=np.int32)

	# run BA algo 
	degree_list = np.array([])
	for i in range(num_trials):

		tic = time.clock()
		# jit this to make faster 
		degrees = add_nodes(nodes,edges,m,num_added)
		toc = time.clock()
		# print the runtime 
		# print 'trial runtime is ' + str(toc - tic)

		degree_list = np.append(degree_list,degrees).flatten()

	# save degree distro data to json 
	if save == True:
		file_path = '../DATA/degrees_' + str(int(np.log10(num_added))) + '_' + str(m) + '_' + str(num_trials) + '.json'
		with open(file_path, 'w') as fp:
				json.dump(degree_list.tolist(), fp) 

	return degree_list

