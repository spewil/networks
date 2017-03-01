from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
import json 
import time
from log_bin import *

# number of nodes added / time steps 
# the number of edges added each time 
# save the figure 
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

		G.add_edges_from([(num_nodes-1,k) for k in rand_nodes])

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
		file_path = 'DATA/degrees_' + str(num_added) + '_' + str(m) + '.json'
		with open(file_path, 'w') as fp:
				json.dump(degree_list, fp) 

	return degree_list


def linearize_order2():

	# first approximation C using guess and check

	C = 8.6
	print 'first guess for a is ' + str(a0)
	err = 1e5
	step = 1e-4
	LogL = np.log10(L)
	num_trials = 500

	# figuring out C iteratively by minimizing the second order fit parameter 
	# make a list of a0's we want to look at
	# start at one or the last value for the first iteration will be zero 
	# (it's exactly our guess) 
	
	a_trials = [x*step + a0 for x in range(1,num_trials+1)]
	loop = 0 
	for a in a_trials:	
		loop += 1
		Log_hOverLa0 = np.log10(1 - (np.array(h_means)/(np.array(L)*a)))
		# print 'current a0 iteration: ' + str(a) 
		# print 1 - (np.array(h_means)/(np.array(L)*a))

		# second order fit  
		coeff_a = np.polyfit(LogL,Log_hOverLa0, 2)

		# plot every 10th iteration 
		if loop%10 == 0:
			ax5.loglog(10**LogL,10**Log_hOverLa0, 'b')

		# take the magnitude of the second derivative
		# we want to minimize this! 
		check = np.absolute(coeff_a[0])
		# highest power first 
		if check < err:
			# update quadratic "error"
			err = check 
			# update our choice of a0 with the quadratically-minimized version 
			# (the current iteration)
			a0 = a
			w1 = coeff_a[1] # the slope
			b = coeff_a[2] # the intercept 
			# print 'current error: ' + str(err)

	a1 = 10**b
	print 'a0 is ' + str(a0)
	print 'a1 is ' + str(a1)
	print 'w1 is ' + str(w1) 

	# plot the "real" one red -- the best approx 
	ax5.loglog(10**LogL,(1 - (np.array(h_means)/(np.array(L)*(a0)))),'r')
	ax5.legend(['Iterated Trials','Minimum Quadratic Coefficient'])
	leg = ax5.get_legend()
	leg.legendHandles[0].set_color('blue')
	leg.legendHandles[1].set_color('red')


if __name__ == '__main__': 
	
	# params 
	m = 1
	num_added = int(5e3)

	run = True 

	if run == True:
		# run BA algo 
		degree_list = run_BA_algo(m,num_added)
	else:
		file_path = 'DATA/degrees_' + str(num_added) + '_' + str(m) + '.json'
		with open(file_path) as fp:
			degree_list = np.array(json.load(fp))


####### RAW DISTRIBUTION #######
	
	# set up figure 
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	# linearly binning 
	prob, degree = lin_bin(degree_list, int(max(degree_list)))
	x = np.linspace(2.5,160,250)
	ax2.plot(prob,degree,'ko',markerSize=1)
	ax2.plot(x,8.7*x**(-3),'r--',lineWidth=.5)
	ax2.legend(['Node Degree Data','Naive Theoretical Solution'])

	print np.min(degree_list)


####### BINNED DATA ########

	# set up figure 
	fig1 = plt.figure()
	ax1= fig1.add_subplot(111)

	# binning plots 
	counts, vals = lin_bin(degree_list,int(max(degree_list)))
	ax1.loglog(counts,vals,'o')
	x, y = log_bin(degree_list,a=1.5)
	ax1.loglog(x,y)

	x = np.linspace(1,max(degree_list),1e3)
	# ax1.loglog(x,theory,'--')

######## SAVING ########

	# binned data 
	fig_file = 'figs/degreedistro_' + str(num_added) + '_' + str(m) + '.pdf'
	fig1.savefig(fig_file)

	# raw distribution 
	fig_file = 'figs/rawdistro_' + str(num_added) + '_' + str(m) + '.pdf'
	fig2.savefig(fig_file)

