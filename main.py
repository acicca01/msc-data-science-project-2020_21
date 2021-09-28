import os
import random
from hRAC.code import code
from hRAC.sparse_indices import sparse_indices
from hRAC.eval import eval
from hRAC.sparsify import sparsify
from hRAC.fetch_mel import fetch_mel
import random
import math
import pandas as pd
import numpy as np
import pickle
from implicit.lmf import LogisticMatrixFactorization
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.sparse
from numpy.random import default_rng
import time
root = "/Users/pantera/melon/arena_mel/"

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
#seed = rng.integers(100000000)
seed = 98831045
with open (os.path.join(files,"seed.pickle"),'wb') as handle:
    pickle.dump(seed,handle)

#mask interactions
sparsify_out = sparsify(testdata,10,seed)
print("Sparsify using seed {} ".format(seed) )
print()

#popularity stats about dropped tracks 
sampled_track = []
for ele in sparsify_out["trackdropped"]:
    sampled_track.append(popularity[ele])

spop = pd.Series(sampled_track)
print("Stats about dropped tracks"
        )
print()
print("--------------------------")
print(spop.describe())

#activity stats about affected users
sampled_user = []
for ele in sparsify_out["userdropped"]:
    sampled_user.append(activity[ele])

sact = pd.Series(sampled_user)

print()
print("Stats about affected users"
        )
print()
print("--------------------------")
print(sact.describe())
print()

#get parameters to build test and train sparse matrices of interactions
train_idx = sparse_indices(sparsify_out['traindata'])
test_idx = sparse_indices(testdata)

#build interaction matrixes
M = len(allplaylists) 
N = len(alltracks)
train_sparse  = csr_matrix((train_idx[0], (train_idx[1], train_idx[2])),shape=(M,N))
test_sparse  = csr_matrix((test_idx[0], (test_idx[1], test_idx[2])),shape=(M,N))
#save sparse matrixes for later evaluation step
scipy.sparse.save_npz(os.path.join(files,"train_sparse_{}".format(seed)),train_sparse)
scipy.sparse.save_npz(os.path.join(files,"test_sparse_{}".format(seed)),test_sparse)

#for model fitting I need to pass the transposed of user-item activity matrix
logmodel = LogisticMatrixFactorization(factors=98, iterations=40, regularization=1.5)
logmodel.fit(train_sparse.T)
print()
print("--------------------------")
print("Saving user & item factors")
#save user-item factors for later evaluation
np.save(os.path.join(files,"item_factors_{}".format(seed)),logmodel.item_factors)
np.save(os.path.join(files,"user_factors_{}".format(seed)),logmodel.user_factors)

#export indexes for popular tracks 
#trackrank = [(i,k) for i,k in enumerate(popularity)]
popularity_train = [len(x) for x in sparsify_out["traindata"]]
trackrank = [(i,k) for i,k in enumerate(popularity_train)]
trackrank.sort(key=lambda x :x[1],reverse = True)
tracks_desired = sparsify_out["trackdropped"]

#building CNN training set
cutoff = 0
for ele in trackrank:
    cutoff+=1
    if ele[1] == 100:
        break
traincnn = []
i=cutoff
while len(traincnn) < 8000:
    if trackrank[i][0] not in tracks_desired:
        traincnn.append(trackrank[i][0])
    i+=1


#building CNN validation set
valcnn = []
i = cutoff + 8001
while len(valcnn) < 2000:

    if trackrank[i][0] not in tracks_desired:
        valcnn.append(trackrank[i][0])
    i+=1

#saving train and validation sets files (including latent factors) to disk for CNN
audio,trainidx = fetch_mel(traincnn)
tmp = map(np.load,audio)
trainset = np.stack(list(tmp))
np.save(os.path.join(files,"trainset{}.npy".format(seed)),trainset)
xport_factors = []
for ele in trainidx:
    xport_factors.append(traincnn_factors[ele])
xport_factors_np = np.asarray(xport_factors)
np.save(os.path.join(files,"trainset_factors{}.npy".format(seed)),xport_factors_np)


audio,validx = fetch_mel(valcnn)
tmp = map(np.load,audio)
valset = np.stack(list(tmp))
np.save(os.path.join(files,"valset{}.npy".format(seed)),valset)
xport_factors = []
for ele in validx:
    xport_factors.append(traincnn_factors[ele])
xport_factors_np = np.asarray(xport_factors)
np.save(os.path.join(files,"valset_factors{}.npy".format(seed)),xport_factors_np)


#batch-exporting mel-spectrograms for CNN testing
tracks_to_test = [x for x in tracks_desired if x not in valcnn+traincnn]
dim = divmod(len(tracks_to_test),5000)
partition = []
for i in range(0,dim[0]):
    if i < dim[0]-1:
        partition.append(tracks_to_test[i*5000:(i+1)*5000])
    else:
        partition.append(tracks_to_test[i*5000:(i*5000+dim[1])])
i = 0
testidx = []
for ele in partition:
    audio,tmpidx = fetch_mel(ele)
    tmp = map(np.load,audio)
    testset = np.stack(list(tmp))
    np.save(os.path.join(files,"testset_{}_{}.npy".format(i,seed)),testset)
    testidx = testidx + tmpidx
    #exporting testset tracks latent factors for CNN evaluation
    xport_factors = []
    for ele in tmpidx:
        xport_factors.append(item_factors[ele])
    xport_factors_np = np.asarray(xport_factors)
    np.save(os.path.join(files,"testset_factors_{}_{}.npy".format(i,seed)),xport_factors_np)
    i+=1
    

#exporting testidx track identifiers we wish to replace their track factors for
with open (os.path.join(files,"testidx{}.pickle".format(seed)),'wb') as handle:
    pickle.dump(testidx,handle)

#exporting test users -- only those very active ( 100+ interactions)
active_users = []
for user in  sparsify_out["userdropped"]:
    if activity[user] > 100:
        active_users.append(user)
with open (os.path.join(files,"testusers{}.pickle".format(seed)),'wb') as handle:
    pickle.dump( active_users,handle)

#exporting popularity in testdata 
with open (os.path.join(files,"testpop{}.pickle".format(seed)),'wb') as handle:
    pickle.dump(popularity ,handle)



#x = eval(train_sparse,test_sparse,logmodel.user_factors,logmodel.item_factors,sparsify_out["userdropped"])

def debug():
    #exporting compressed audio
    pca = PCA(n_components =3 )
    scaler = StandardScaler()
    audio_files = []
    for i in range(len(traincnn_decoded)):
        audio_files.append(os.path.join(root,str(math.floor(traincnn_decoded[i]/1000)),str(traincnn_decoded[i])+".npy"))

    def compress(x):
        tmp = np.load(x)
        return pca.fit_transform(tmp)
    tmp = map(compress,audio_files)
    trainset_compress = np.stack(list(tmp))
    np.save(os.path.join(files,"trainset_compress_{}.npy".format(seed)),trainset_compress)

    #exporting for scaled audio
    def scale(x):
        tmp = np.load(x)
        return scaler.fit_transform(tmp)
    tmp = map(scale,audio_files)
    trainset_scale = np.stack(list(tmp))
    np.save(os.path.join(files,"trainset_scale_{}.npy".format(seed)),trainset_scale)


