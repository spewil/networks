from __future__ import division 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
from scipy.optimize import curve_fit
import json 
import time
from log_bin import *
from collections import Counter
from BAmodel3 import *


if __name__ == '__main__': 
	
	# params 
	m = [3]#[1,2,3]
	num_added = [1e4]#[1e2,1e3,1e4]
	num_trials = 100
	
	# run or open data from file 
	run = 1
	save = True

		# runs trials and saves total jsons 
		# tic = time.clock()
	for i in range(len(num_added)):
		for j in range(len(m)):		

			if run == True:
				tic = time.clock()
				degree_list = run_BA3(m[j],num_added[i],num_trials,save)
				toc = time.clock()
				print 'total runtime is ' + str(toc - tic)

			else:
				file_path = '../DATA/degrees_' + str(int(np.log10(num_added[i]))) + '_' + str(m[j]) + '_' + str(num_trials) + '.json'
				with open(file_path) as fp:
					degree_list = np.array(json.load(fp))

			print 'number of total nodes: ' + str(len(degree_list))
			# print str(100*(1000+7) - (1000+7)*m[j])

		####### RAW DISTRIBUTION #######
			
			# set up figure 
			fig2 = plt.figure()
			ax2 = fig2.add_subplot(111)
			# linearly binning 
			# prob, degree = lin_bin(degree_list, int(max(degree_list)))

			# degree=range(int(np.min(degree_list)),int(np.max(degree_list))+1)	 	
			# prob = []
			# k is the degree 
			# for k in degree:
			# 	# True/False masking for the degrees, if it is degree k, make True
			# 	mask = (degree_list == k) 
			# 	# normalize by the number of discrete probabilities 
			# 	prob.append(len(degree_list[mask])/len(degree_list))

			# number of unique degrees
			# prob = number of nodes of degree k vs k 
			freqs = Counter(degree_list)
			degree = freqs.keys()
			prob = [f/sum(freqs.values()) for f in freqs.values()]
			# prob = [f/sum(degree_list) for f in degree_list]

			ax2.plot(degree,prob,'ko',markerSize=2,zorder=100)
			x = np.linspace(2,100,1000)
			ax2.plot(x,(m[j]*(m[j]+1)/2)*x**(-3),'r--',lineWidth=1)

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
			ax3.loglog(x,(2*m[j]*(m[j]+1))*x**(-3),'r--',lineWidth=1)
			# the naive numerical model 
			# ax3.loglog(x,naive_theory(x,popt[0],popt[1]),'b--',lineWidth=1)
			ax3.loglog(x,(2*m[j]*(m[j]+1))/(x*(x+1)*(x+2)),'o--',lineWidth=1)

			ax3.legend(['Log-Binned Data','Naive Theoretical Solution','Exact Theoretical Solution'])	


		####### BINNED DATA ########

			# set up figure 
			fig1 = plt.figure()
			ax1= fig1.add_subplot(111)

			# ax1.loglog(degree,prob,'ko',markerSize=.5)
			# binning plots 
			#log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):
			centers, counts = log_bin(degree_list,bin_start=m[j],a=1.5)
			ax1.loglog(centers,counts)

			# naive guess 
			x = np.linspace(1,max(degree_list),1e3)
			ax1.loglog(x,(2*m[j]*(m[j]+1))*x**(-3),'r--',lineWidth=1)

			# exact long time solution 
			ax1.loglog(x,(2*m[j]*(m[j]+1))/(x*(x+1)*(x+2)),'o--',lineWidth=1)

			ax1.legend(['Node Degree Data','Naive Theoretical Solution','Exact Theoretical Solution'])	


		######## SAVING ########

			# binned data 
			fig_file = '../figs/degreedistro_' + str(int(np.log10(num_added[i]))) + '_' + str(m[j]) + '_' + str(num_trials) + '.pdf'
			fig1.savefig(fig_file)
			# raw distribution 
			fig_file = '../figs/rawdistro_' + str(int(np.log10(num_added[i]))) + '_' + str(m[j]) + '_' + str(num_trials) + '.pdf'
			fig2.savefig(fig_file)
			# loglog data  
			fig_file = '../figs/logdistro_' + str(int(np.log10(num_added[i]))) + '_' + str(m[j]) + '_' + str(num_trials) + '.pdf'
			fig3.savefig(fig_file)



