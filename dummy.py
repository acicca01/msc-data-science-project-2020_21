# Create dummy music collection
# Music collection will consist of 10 playlist featuring 10 songs each. Total song in collection is 14

import pandas as pd
import numpy as np

# create playlist identifiers
plylst_id = np.arange(0,10,1)

# create 14 random weights 
weights = np.random.choice(66,14)
somma = sum(weights)

# probs is a list of probabilities, with prob[i] = probability i-th song of the collection is selected for a playlist 
probs=[]
for ele in weights:
    probs.append(ele/somma)

mysongs=[]
for i in range(10):
    mysongs.append(np.random.choice(14,10, p = probs, replace = False))

dummy = pd.DataFrame()
dummy["plylst_id"] = pd.Series(plylst_id)
dummy["songs"] = pd.Series(mysongs)




