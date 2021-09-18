import os
from hRAC.code import code
from hRAC.sparse_indices import sparse_indices
from hRAC.eval import eval
from hRAC.sparsify import sparsify
import random
import math
import pandas as pd
import numpy as np
import pickle
from implicit.lmf import LogisticMatrixFactorization
from scipy.sparse import csr_matrix
from numpy.random import default_rng
import time


#location for intermediate files drop. Useful when switching environment for CNN training
files = "/Users/pantera/melon/files"

#read in the data and discover playlists-track interaction
train = pd.read_json('/Users/pantera/melon/train.json')
train2 = train[['id','songs']]
train3 = train2.assign(tracks_count = train2['songs'].apply(lambda x: len(x)))
train3 = train3.rename(columns={"songs":"tracks"})
tracks_dict={train3["tracks"][i][j] : [] for i in range(train3.shape[0]) for j in range(train3["tracks_count"][i])}

#this is the step loads the interations into a dictionary 
# {track_ID : [playlist_ID of playlists playing track_ID] }
#it's very resource hungry hence run only once and saved
#train3.apply(lambda row: discover(row['tracks'],row['id'],tracks_dict) , axis = 1)

#with open('/Users/pantera/melon/tracksdict.pickle', 'wb') as handle:
#    pickle.dump(tracks_dict, handle)

#read in the dictionary of interactions
with open('/Users/pantera/melon/tracks_dict.pickle', 'rb') as handle:
    tracks_dict = pickle.load(handle)


#encode dictionary entries (i.e. mapping the set of track/PlayList IDs to sequential IDs)
alltracks = [x for x in tracks_dict.keys()]
allplaylists = [tracks for playlist in tracks_dict.values() for tracks in playlist]
allplaylists=list(set(allplaylists))
trackmap,decode = code(alltracks,"tracks")
playlistmap = code(allplaylists,"playlists")[0]
tracks_dict_map={trackmap[k]: [playlistmap[pl_id] for pl_id in v] for k ,v in tracks_dict.items()}


#Encoding original "train" dataframe for reference
train_coded = train3.assign(id_encoded = train3["id"].map(playlistmap))
train_coded = train_coded.assign(tracks_encoded = train_coded["tracks"].map(lambda v : [trackmap[x] for x in v]))
plays_dict_map = {train_coded['id_encoded'][i] : train_coded['tracks_encoded'][i]  for i in range(train_coded.shape[0]) }

#now that we have sequential IDs for playlists and tracks we can store interactions as arrays

#index i represent a trackID, testdata[i] is a list of playlist IDs the track interacts with
testdata = list(range(len(alltracks)))

#testdata will contain all original interactions
for k,v in tracks_dict_map.items():
    testdata[k] = v

#userdata to check user activity. Less active users will be dropped
userdata = list(range(len(allplaylists)))
for k,v in plays_dict_map.items():
    userdata[k] = v

#get user activity
activity = [len(x) for x in userdata]
#get tracks popularity
popularity = [len(x) for x in testdata]

rng = default_rng()
seed = rng.integers(100000000)
sparsify(testdata,10,seed)
#mask interactions
sparsify_out = sparsify(testdata,10,seed)
print("Sparsify using seed {} ".format(seed) )
print()

#popularity stats about dropped tracks 
sampled_track = []
for ele in sparsify_out["trackdropped"]:
    sampled_track.append(popularity[ele])

spop = pd.Series(sampled_track)
print(spop.describe())

#activity stats about affected users
sampled_user = []
for ele in sparsify_out["userdropped"]:
    sampled_user.append(activity[ele])

sact = pd.Series(sampled_user)
print(sact.describe())

#get parameters to build hot and cold sparse matrices of interactions
train_idx = sparse_indices(sparsify_out['traindata'])
test_idx = sparse_indices(testdata)
#build interaction matrixes
M = len(allplaylists) 
N = len(alltracks)
train_sparse  = csr_matrix((train_idx[0], (train_idx[1], train_idx[2])),shape=(M,N))
test_sparse  = csr_matrix((test_idx[0], (test_idx[1], test_idx[2])),shape=(M,N))

#for model fitting I need to pass the transposed of user-item activity matrix
logmodel = LogisticMatrixFactorization(factors=32, iterations=40, regularization=1.5)
logmodel.fit(train_sparse.T)


#export indexes for popular tracks 
trackrank = [(i,k) for i,k in enumerate(popularity)]
trackrank.sort(key=lambda x :x[1],reverse = True)
traincnn=[]
for ele in trackrank:
    if ele[1] == 100:
        break
    else:
        traincnn.append(ele[0])
print("Extracted top {}  popular tracks for CNN".format(len(traincnn)))

#decode trackids to allow proper retrieval of audio files
traincnn_decoded  = [decode[x] for x in traincnn]
with open('/Users/pantera/melon/files/traincnn_decoded.pickle', 'wb') as handle:
    pickle.dump(traincnn_decoded, handle)
#export factors for popular tracks to be used as class variables for CNN training

np.save(os.path.join(files,"traincnn_factors.npy"),logmodel.item_factors[traincnn])
#x = eval(train_sparse,test_sparse,logmodel.user_factors,logmodel.item_factors,sparsify_out["userdropped"])


