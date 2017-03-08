from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from scipy.optimize import curve_fit
import json 
import time
from log_bin import *
from collections import Counter
from BAmodel import *

if __name__ == '__main__': 
	
	# params 
	m = 1
	num_added = int(1e3)
	num_trials = 100
	
	# run or open data from file 
	run = 1 
	save = True

	if run == True:
		# runs trials and saves total jsons 
		tic = time.clock()
		degree_list = run_BA(m,num_added,num_trials,save)
		toc = time.clock()
		print 'total runtime is ' + str(toc - tic)

	else:
		file_path = '../DATA/degrees_' + str(int(np.log10(num_added))) + '_' + str(m) + '_' + str(num_trials) + '.json'
		with open(file_path) as fp:
			degree_list = np.array(json.load(fp))

	add_nodes.inspect_types()

####### RAW DISTRIBUTION #######
	
	# set up figure 
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	# linearly binning 
	# prob, degree = lin_bin(degree_list, int(max(degree_list)))

	degree=range(int(np.min(degree_list)),int(np.max(degree_list))+1)	 	
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

	ax2.legend(['Node Degree Data','Naive Theoretical Solution'])#,'Naive Numerical Fit'])	

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

	ax3.legend(['Node Degree Data','Naive Theoretical Solution'])#,'Naive Numerical Fit'])	


####### BINNED DATA ########

	# set up figure 
	fig1 = plt.figure()
	ax1= fig1.add_subplot(111)

	ax1.loglog(degree,prob,'o')
	# binning plots 
	# def log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):
	x, y = log_bin(degree_list,a=1.7)
	ax1.loglog(x,y)

	# naive guess 
	x = np.linspace(1,max(degree_list),1e3)
	ax1.loglog(x,9*x**(-3),'r--',lineWidth=1)

######## SAVING ########

	# binned data 
	fig_file = '../figs/degreedistro_' + str(int(np.log10(num_added))) + '_' + str(m) + '_' + str(num_trials) + '.pdf'
	fig1.savefig(fig_file)
	# raw distribution 
	fig_file = '../figs/rawdistro_' + str(int(np.log10(num_added))) + '_' + str(m) + '_' + str(num_trials) + '.pdf'
	fig2.savefig(fig_file)
	# loglog data  
	fig_file = '../figs/logdistro_' + str(int(np.log10(num_added))) + '_' + str(m) + '_' + str(num_trials) + '.pdf'
	fig3.savefig(fig_file)



