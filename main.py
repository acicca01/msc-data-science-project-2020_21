import pandas as pd
import numpy as np
import pickle
import implicit
from scipy.sparse import csr_matrix
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
trackmap = code(alltracks)
playlistmap = code(allplaylists)
tracks_dict_map={trackmap[k]: [playlistmap[pl_id] for pl_id in v] for k ,v in tracks_dict.items()}

#get popularity
popularity = {x : len(v) for x,v in tracks_dict_map.items() }

#split according to popularity
cold = {}
hot = {}
for k,v in tracks_dict_map.items():
    if popularity[k] > 20:
        hot[k] = v
    else:
        cold[k] = v

#get parameters to build hot and cold sparse matrices of interactions
hot_idx = sparse_indices(hot)
cold_idx = sparse_indices(cold)

#build interaction matrixes
M = len(allplaylists) 
N = len(alltracks)
hot_sparse  = csr_matrix((hot_idx[0], (hot_idx[1], hot_idx[2])),shape=(M,N))
cold_sparse = csr_matrix((cold_idx[0], (cold_idx[1], cold_idx[2])),shape=(M,N))



#implicit 


#evaluation



