from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from scipy.optimize import curve_fit
import json 
import time
from log_bin import *
from collections import Counter
# from numba import jit 

# number of nodes added / time steps 
# the number of edges added each time 
# save the figure 
# @jit 
def add_BA_nodes(G,num_added_nodes,m,save=True,filename='net.pdf'):

	for i in range(num_added_nodes):

		# empty list of randomly chosen nodes 
		rand_nodes = []

		# grab the list of nodes, edges, degrees
		degrees = np.array(G.degree().values())

		# num of total edges (double counting)
		num_edges_tot = np.sum(degrees)
		
		# list of probabilities node_degree/total edges 
		probs = degrees/num_edges_tot

		print sum(probs)

		# make a nonuniform choice of m nodes from the list of nodes
		# before adding a node! to protect against self edges 
		# probability based linearly on degree, P(k) ~ k 
		# returns ndarray([node_indices])
		rand_nodes = np.random.choice(G.nodes(), m, replace=False, p=probs)

		num_nodes = len(G.nodes())
		# adds a node with index at the end 
		G.add_node(num_nodes)
		# update length 
		num_nodes = len(G.nodes())

		nodes_to_add = [(num_nodes-1,k) for k in rand_nodes]
		G.add_edges_from(nodes_to_add)

	nx.draw_spring(G,node_size=5) # networkx draw()
	if save == True:
		net.savefig(filename)


def run_BA_algo(m,num_added, save=True): 
	# generate a starting graph 
	# 50 nodes to start 
	# complete graph to start, p_edge = .3
	# want at least the number to connect to 
	G=nx.complete_graph(m+1)

	tic = time.clock()
	add_BA_nodes(G,num_added,m,'mynet.pdf')
	toc = time.clock()
	# print the runtime 
	print 'runtime is ' + str(toc - tic)

	# grab the list of degrees 
	degree_list = G.degree().values()

	# save degree distro data to json  
	if save == True:
		file_path = '../DATA/degrees_' + str(int(np.log10(num_added))) + '_' + str(m) + '.json'
		with open(file_path, 'w') as fp:
				json.dump(degree_list, fp) 

	return degree_list


if __name__ == '__main__': 
	
	# params 
	m = 2
	num_added = int(1e4)
	num_experiments = 10
	run = True 

	if run == True:
		# run BA algo 
		degree_list = []
		for i in range(num_experiments):
			degree_list.append(run_BA_algo(m,num_added,save=True)).flatten()

	else:
		file_path = '../DATA/degrees_' + str(int(np.log10(num_added))) + '_' + str(m) + str(num_experiments) + '.json'
		with open(file_path) as fp:
			degree_list = np.array(json.load(fp))

	print len(degree_list)

####### RAW DISTRIBUTION #######
	
	# set up figure 
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	# linearly binning 
	# prob, degree = lin_bin(degree_list, int(max(degree_list)))

	degree=range(np.min(degree_list),np.max(degree_list)+1)	 	
	prob = []
	# k is the degree 
	# for k in degree:
	# 	# True/False masking for the degrees, if it is degree k, make True
	# 	mask = (degree_list == k) 
	# 	# normalize by the number of discrete probabilities 
	# 	prob.append(len(degree_list[mask])/len(degree_list))

	# words = "apple banana apple strawberry banana lemon"
	freqs = Counter(degree_list)#.split())
	degree = freqs.keys()
	prob = [i/sum(freqs.values()) for i in freqs.values()]
	print(sum(prob))

	ax2.plot(degree,prob,'ko',markerSize=2,zorder=100)
	x = np.linspace(2,100,1000)
	ax2.plot(x,9*x**(-3),'r--',lineWidth=1)

	# # the naive theoretical function 
	# def naive_theory(x, a, b):
	#     return a*(x**-3) + b 
	# popt, pcov = curve_fit(naive_theory, degree, prob, bounds=([5.,-1], [10., 1.]))
	# print 'covariance matrix is ' + str(pcov)
	# print 'a is ' + str(popt[0]) + ' and b is ' + str(popt[1])
	# ax2.plot(x,naive_theory(x,popt[0],popt[1]),lineWidth=1)

	ax2.legend(['Node Degree Data','Naive Theoretical Solution'])

	# print np.min(degree_list)

####### LOGGED DATA ########

	# set up figure 
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	# data 
	ax3.loglog(degree,prob,'ko',markerSize=2,zorder=100)
	x = np.linspace(2,100,1000)
	# naive guess 
	ax3.loglog(x,9*x**(-3),'r--',lineWidth=1)
	# the naive numerical model 
	# ax3.loglog(x,naive_theory(x,popt[0],popt[1]),'b--',lineWidth=1)

	ax3.legend(['Node Degree Data','Naive Theoretical Solution','Naive Numerical Fit'])	


####### BINNED DATA ########

	# set up figure 
	fig1 = plt.figure()
	ax1= fig1.add_subplot(111)

	ax1.loglog(degree,prob,'o')
	# binning plots 
	# def log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):
	x, y = log_bin(degree_list,bin_start=np.min(degree_list),first_bin_width=1,a=1.7)
	ax1.loglog(x,y)

	# naive guess 
	x = np.linspace(1,max(degree_list),1e3)
	ax1.loglog(x,9*x**(-3),'r--',lineWidth=1)

######## SAVING ########

	# binned data 
	fig_file = '../figs/degreedistro_' + str(int(np.log10(num_added))) + '_' + str(m) + '.pdf'
	fig1.savefig(fig_file)
	# raw distribution 
	fig_file = '../figs/rawdistro_' + str(int(np.log10(num_added))) + '_' + str(m) + '.pdf'
	fig2.savefig(fig_file)
	# loglog data  
	fig_file = '../figs/logdistro_' + str(int(np.log10(num_added))) + '_' + str(m) + '.pdf'
	fig3.savefig(fig_file)



