from hRAC.code import code
from hRAC.sparse_indices import sparse_indices
from hRAC.takedown import takedown
import random
import math
import pandas as pd
import numpy as np
import pickle
from implicit.lmf import LogisticMatrixFactorization
from scipy.sparse import csr_matrix
from sklearn import metrics
import time
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


#now we can deal with arrays
traindata = np.asarray([ v for _,v in tracks_dict_map.items() ],dtype=object)
#Encoding original "train" dataframe for reference
train_coded = train3.assign(id_encoded = train3["id"].map(playlistmap))
train_coded = train_coded.assign(tracks_encoded = train_coded["tracks"].map(lambda v : [trackmap[x] for x in v]))
plays_dict_map = {train_coded['id_encoded'][i] : train_coded['tracks_encoded'][i]  for i in range(train_coded.shape[0]) }
#get popularity
popularity = [len(x) for x in traindata]

#masking cold start tracks (i.e. less than 30 interactions) 
#for those tracks drop 50% of the interactions at random 
#!! cold start tracks with only 1 interaction will be spared from dropping

#testdata = traindata.copy()
mask = [0 < interactions < 31 for interactions in popularity]
wipepop = [31<interactions for interactions in popularity]


traindata = np.asarray([ v for _,v in tracks_dict_map.items() ],dtype=object)
for i in range(len(traindata)):
    if wipepop[i] == True:
        traindata[i] = []
testdata = traindata.copy()


for i in range(len(traindata)):
    if mask[i] == True:
        traindata[i] = random.sample(traindata[i],int(len(traindata[i])*0.5))

#check percentages of interactions masked
#interaction of j-th column i-th row . Column = tracks , row = playlist
#interactionstrain = ( (j,i) for j in range(len(traindata)) for i in traindata[j])
#interactionstest = ( (j,i) for j in range(len(testdata)) for i in testdata[j])
#(1 - (len([x for x in interactionstrain])/len([x for x in interactionstest])))*100

#build a list so that masked_playlists[i] gives the tracks that should be recommended to playlist i
masked_playlists = [ [] for i in range(len(allplaylists))]
for i in range(len(traindata)):
    if mask[i] == True:
        for j in range(len(testdata[i])):
            # testdata[i][j] gives me playlist id being masked
            if testdata[i][j] not in traindata[i]:
                masked_playlists[testdata[i][j]].append(i)

#for each playlist give me tracks exposed to training data
exposed_playlists = [ [] for i in range(len(allplaylists))]                
for i in range(len(traindata)):
    for j in range(len(traindata[i])):
        exposed_playlists[traindata[i][j]].append(i)

#get parameters to build hot and cold sparse matrices of interactions
train_idx = sparse_indices(traindata)
test_idx = sparse_indices(testdata)
#build interaction matrixes
M = len(allplaylists) 
N = len(alltracks)

#for model fitting I need to pass the transposed of user-item activity matrix
train_sparse  = csr_matrix((train_idx[0], (train_idx[1], train_idx[2])),shape=(M,N))
#leaving test data as it is (user-items)
test_sparse  = csr_matrix((test_idx[0], (test_idx[1], test_idx[2])),shape=(M,N))
#implicit 
logmodel = LogisticMatrixFactorization(factors=30, iterations=40, regularization=1.5)
logmodel.fit(train_sparse.T)

#evaluation
#filter out users who did not have any item masked
modplaylist = [i for i in range(len(masked_playlists)) if len(masked_playlists[i]) != 0]
start = time.time()
auc_scores = []
for i in modplaylist:
    actual = test_sparse[i,:].toarray().reshape(-1)
    actual_less = np.delete(actual,exposed_playlists[i])
    scores = logmodel.user_factors[i,:].dot(logmodel.item_factors.T)
    scores_less = np.delete(scores,exposed_playlists[i])
    fpr, tpr, thresholds = metrics.roc_curve(actual_less, scores_less)
    auc_scores.append(metrics.auc(fpr, tpr))
print("generated scores in {} minutes ".format(int(time.time() -start)/60))





#compare lists
def cl(base,compare):
    match = 0
    for ele in compare:
        for ele2 in base:
              if ele == ele2:
                  match+=1
    return match
