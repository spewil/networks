'''
Created on 14 Jan 2014
This creates ER random graphs for fixed N and a series of different <k> values
@author: time
'''

# Useful for manipulating filenames without worrying about which OS you use
import os

# Very useful network library
import networkx as nx

# matplotlib plotting library produces excellent high quality plots.
# It emulates MatLab (which may or may not be of use to you).
# I would think of using python/matplotlib for plotting even if you do your sums in another language.
# For plot tutorial see http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt

import numpy as np

# I find that UNIX style forward slashes in directory and file names work on any OS in most languages.
# They should be backwards slashes on Windows machines but a backwards slash has a special meaning
# in strings.  So for Windows users I recommend just using a forwards slash where a backwards would be used on Windows.
#
# This is the directory where I want my outputs to go
# MAKE SURE DIRECTORY EXISTS!
# outputdir='h:/CandN/output/"
outputdir="/DATA"

# Number of nodes in the graph.  Try 10000 if you want good data, 30 if you want to visualise
N=20

# The names of all output files will start with this
filenameroot="er_N"+str(N)

# min, max and step size for sequence of <k> values to look at
# NOTE a common problem is that python will assume integer arithmetic if you define constants as integers.
# I recommend you use a floating poin notation if you meant the number to be a float e.g. 0.0 or 1.0 not 0 or 1
kmin=0.0
kmax=2.0
kstep=0.05

elllist=[] # record ell values
k=kmin
while k<=kmax:
    p=k/(N-1) 
    g=nx.erdos_renyi_graph(N,p) # make ER graph
    filename=filenameroot+'_k'+str(int(k*1000+0.5))+".net"
    outputfile=os.path.join(outputdir,filename) # os module is best way to deal with file names
    print "Writing network to ",outputfile
    nx.write_pajek(g, outputfile) # write out .net file, comment out if don't want this
    elltotal=0
    npaths=0
    componentlist= list(nx.connected_component_subgraphs(g))
    for c in componentlist:
        nc=c.order()    #Return the number of nodes in the graph.
        if nc<2:  # only want components with two nodes or more
            continue   
        try: 
            ellc=nx.average_shortest_path_length(c) 
            thesepaths=nc*(nc-1)/2
            npaths=npaths+thesepaths
            elltotal=ellc*thesepaths+elltotal   
        except: # something went wrong if reached here, disconnected graph?
            ellc=-1
        #print "nc,ellc=",nc,ellc
    
    ell=0
    if npaths>0:
        ell=elltotal/npaths    
    print "p=",p,", k=",k,", <l>=",ell
    # lc = sum(1 for _ in componentlist)
    elllist.append([p,k,ell, len(componentlist), componentlist[0].order()])
    #p=p+pstep
    k=k+kstep 

# now write file of data
# numpy has some built in ways to do this but here is a 
filename=filenameroot+'_kmin'+str(int(kmin*1000+0.5))+'_kmax'+str(int(kmax*1000+0.5))+'_kstep'+str(int(kstep*1000+0.5))+".dat"
fullfilename=os.path.join(outputdir,filename)
try:
    # I always print out the file name as I am always having problems with directories etc
    print "--- opening data file  ",fullfilename 
    f=open(fullfilename,"w")
    # note backwards slash indicates special character \t = tab, \n = newline
    # tab separated text files are usually best when reading into other packages e.g. excel
    f.write('p\tk\tell\t\tncomp\tgcc\n') # this is the header line
    for ddd in elllist:
        try:
            # now write one line of data 
            f.writelines('%f\t%f\t%f\t%i\t%i\n'%(ddd[0],ddd[1],ddd[2],ddd[3],ddd[4]))
            #f.write('{0}\t{1}\t{4}\t{3}\n'.format(ddd[0],ddd[1],ddd[2],ddd[3]))
        except:
            pass 
    print "--- finished writing data file  ",fullfilename
    f.close()   
except:
    print "*** Failed to finish data file  ",fullfilename  
    
# Now plot answers out
    
transposed = []
for i in range(5):
    transposed.append([row[i] for row in elllist])

# transposed[0] = probability p
# transposed[1] = average path length between connected vertex pairs
# transposed[2] = number of components
# transposed[3] = number of vertices in GCC




# .........................
plt.figure() # start a new plot and give it an number, starts from 0 and increments otherwise
plt.plot(transposed[1],transposed[2])
plt.xlabel(r'$\langle k \rangle$')
plt.ylabel(r'$\ell$')
filenamebasic=filenameroot+"_k_ell"
print "--- plotting k vs ell to  ",filenamebasic 
# see http://matplotlib.org/api/pyplot_api.html?highlight=savefig#matplotlib.pyplot.savefig

ext=".eps"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
ext=".pdf"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
# To display on screen use plt.show() but do this after you output to a file 
#plt.show()

# .........................
plt.figure() 
plt.plot(transposed[1],transposed[3])
plt.xlabel(r'$\langle k \rangle$')
plt.ylabel('$n_{comp}$')
filenamebasic=filenameroot+"_k_ncomp"
print "--- plotting k vs ncomp to  ",filenamebasic 
# see http://matplotlib.org/api/pyplot_api.html?highlight=savefig#matplotlib.pyplot.savefig

ext=".eps"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
ext=".pdf"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
#plt.show()

# .........................
plt.figure() 
plt.plot(transposed[1],transposed[4])
plt.xlabel(r'$\langle k \rangle$')
plt.ylabel('$n_{gcc}$')
filenamebasic=filenameroot+"_k_ngcc"
print "--- plotting k vs ngcc to  ",filenamebasic 
# see http://matplotlib.org/api/pyplot_api.html?highlight=savefig#matplotlib.pyplot.savefig

ext=".eps"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
ext=".pdf"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
#plt.show()



# .........................
plt.figure() 
plt.xlabel(r'$\langle k \rangle$')
ellmax=max(transposed[2])
y= [x/ellmax for x in transposed[2]]
pell, = plt.plot(transposed[1],y, '-', color='r', label='$\ell$')
ncompmax=max(transposed[3])
y= [x/ncompmax for x in transposed[3]]
pncomp, = plt.plot(transposed[1],y, '-', color='b', label='$n_{gcc}$')
ngccmax=max(transposed[4])
y= [x/ngccmax for x in transposed[4]]
pngcc, = plt.plot(transposed[1],y, '-', color='g', label='$n_{gcc}$')
#plt.legend([pell,pncomp,pngcc],['$<k>$','$n_{gcc}$','$n_{gcc}$'])

filenamebasic=filenameroot+"_k_all"
print "--- plotting k vs all to  ",filenamebasic 
# see http://matplotlib.org/api/pyplot_api.html?highlight=savefig#matplotlib.pyplot.savefig

ext=".eps"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
ext=".pdf"
filename=filenamebasic+ext
plotfilename=os.path.join(outputdir,filename)
plt.savefig(plotfilename)
#plt.show()



print "*** Finished plots  "     
