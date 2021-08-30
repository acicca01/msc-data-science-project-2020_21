#import libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import log
import powerlaw



#open dictionary with track-playlist interaction and define popularity as number of playlists listening to track 
with open('/Users/pantera/melon/tracks_dict.pickle', 'rb') as handle:
    tracks_dict = pickle.load(handle)
popularity = {x : len(v) for x,v in tracks_dict.items() }
popdata = np.zeros(len(popularity))
i = 0
for k in popularity.keys():
    popdata[i]=popularity[k]
    i+=1

#function to log space consecutive bin edges 
def logbins(data,n):
    return np.logspace(np.log10(min(data)),np.log10(max(data)),n)

#fit powerlaw model to the data
fit = powerlaw.Fit(popdata)
#display power law in dataset
fig, ax = plt.subplots(figsize = (7,5))
ax.set_ylabel("Track Interactions")
ax.set_xlabel("Number of tracks")
plt.xscale('log')
plt.yscale('log')
ax.tick_params(left = False, bottom = False)
hist, bins, _  = plt.hist(popdata,bins=logbins(popdata,30),facecolor='g', alpha=0.55 
                         ,density = True)
powerlaw.plot_pdf(popdata,color='b')


#fig, ax = plt.subplots(figsize = (7,5))
#ax.set_ylabel("Track Popularity")
#ax.set_xlabel("Number of tracks")
#plt.xscale('log')
#plt.yscale('log')
#ax.tick_params(left = False, bottom = False)
#powerlaw.plot_pdf(popdata,color='b')
#hist, bins, _  = plt.hist(popdata,bins=logbins(popdata,200),facecolor='g', alpha=0.55,density = True )
plt.savefig('/Users/pantera/melon/figures/fig2.1.png')
