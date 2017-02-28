"""
This is an example of using the networkx library for the networks project for C&N

"""

# Very useful network library
import networkx as nx

# Simple random number package
import random

# matplotlib plotting library produces excellent high quality plots.
# It emulates MatLab (which may or may not be of use to you).
# I would think of using python plus matplotlib for plotting even if you do your sums in another language.
# For plot tutorial see http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt

# numpy is full of very useful things, as is scipy, you may need them later
#import numpy as np
#import scipy.stats as stats

# Useful for manipulating filenames without worrying about which OS you use
import os

# total number of vertices required
N=8

# Expectation value for average degree
kexp=4.0

# Will only print out information if N is below this number, use for testing
Nprint=20

# I find that UNIX style forward slashes in directory and file names work on any OS in most languages.
# They should be backwards slashes on Windows machines but a backwards slash has a special meaning
# in strings.  So for Windows users I recommend just using a forwards slash where a backwards would be used on Windows.
#
# This is the directory where I want my outputs to go
# MAKE SURE DIRECTORY EXISTS!
# outputdir='h:/CandN/output/"
outputdir='/DATA/CandN/output/'

# set seed to a number if you want to fix the random number seed for testing
# Otherwise set seed = None if you want true random numbers
#seed=0
seed = None



#
# end of user defined variables
# **********************************

# cutout too much printing
printon = False 
if (N<Nprint):
    printon=True # print stuff when debugging with small N
    

# Random numbers, not used here but in general of use
# see http://docs.python.org/2/library/random.html
# I often use random.randrange, random.sample, and random.random
if seed is None:
    random.seed()
    print ' Random number seed not fixed for proper run'
else:
    random.seed(seed)
    print ' Random number seed fixed for testing'


# The names of all output files will start with this
filenameroot='er_N'+str(N)+'_k'+str(kexp)



# initialise graph
G=nx.Graph() # Undirected Simple graph http://networkx.github.io/documentation/latest/reference/classes.html

# optional
G.name='ER_graph({0:d},{1:.2f})'.format(N,kexp)


#This section add N nodes.
# Each node has unique number starting with 0 up to (N-1) if there are N nodes
for i in range(N):
    G.add_node(i)
# Could have used 
#     G.add_node(nx.number_of_nodes(G))
# which means each time you add a node it is labelled with the next integer,
# with the first node labelled as zero.

# Probability that we add an edge
p=kexp/float(N-1)  

# Now add edges    
# Note that we avoid self-edges
for s in range(N):
    for t in range(s+1,N):
        if (random.random() <p):
            G.add_edge(s,t)    
            if printon:
                print '--- new edge from',s,' to ',t # useful for debugging

# May not need this.
# Now output a pajek .net file.  
# Quite easy to read this file format in a text editor to check all is OK
# Can do visualisation or analysis in other packages such as gephi, visone or pajek 
# as most packages will read this format.
filename=filenameroot+'.net'
outputfile=os.path.join(outputdir,filename) # os module is best way to deal with file names
print "Writing network to ",outputfile
nx.write_pajek(G, outputfile) # write out .net file, comment out if don't want this

        

 