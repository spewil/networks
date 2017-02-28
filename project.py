from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from log_bin import *

# number of nodes added / time steps 
# the number of edges added each time 
# save the figure 
def add_BA_nodes(G,num_added_nodes,m,save=True,filename='net.pdf'):

	for i in range(num_added_nodes):#num_nodes):	

		# empty list of randomly chosen nodes 
		rand_nodes = []

		# grab the list of nodes, edges, degrees
		degrees = np.array(G.degree().keys())

		# num of total edges (double counting)
		num_edges_tot = np.sum(degrees)
		
		# list of probabilities node_degree/total edges 
		probs = degrees/num_edges_tot

		# make a nonuniform choice of m nodes from the list of nodes
		# probability based linearly on degree, P(k) ~ k 
		# returns ndarray([node_indices])
		rand_nodes = np.random.choice(G.nodes(), m, replace=True, p=probs)

		num_nodes = len(G.nodes())
		# adds a node with index at the end 
		G.add_node(num_nodes)
		# update length 
		num_nodes = len(G.nodes())

		G.add_edges_from([(num_nodes-1,k) for k in rand_nodes.flatten()])

	nx.draw_spring(G,node_size=5) # networkx draw()
	if save == True:
		net.savefig(filename)

# generate a starting graph 
# 50 nodes to start 
# ER graph to start, p_edge = .3
G=nx.fast_gnp_random_graph(50,0.3)

add_BA_nodes(G,1000,5,'mynet.pdf')

fig = plt.figure()
ax= fig.add_subplot(111)

degrees = G.degree().values()

counts, vals = log_bin(degrees)
ax.loglog(counts,vals)

fig.savefig('degreedistro.pdf')

