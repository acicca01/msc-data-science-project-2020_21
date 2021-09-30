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

##Set seed before training
#importing seed from main session
with open (os.path.join(files,"seed.pickle"),'rb') as handle:
    seed = pickle.load(handle)

#loading files for evaluation

#sparse matrices
train_sparse = scipy.sparse.load_npz(os.path.join(files,"train_sparse_{}.npz".format(seed)))
test_sparse = scipy.sparse.load_npz(os.path.join(files,"test_sparse_{}.npz".format(seed)))

#latent factors
item_factors = np.load(os.path.join(files,"item_factors_{}.npy".format(seed)))
user_factors = np.load(os.path.join(files,"user_factors_{}.npy".format(seed)))

#ids of tracks to be improve on
with open (os.path.join(files,"testidx{}.pickle".format(seed)),'rb') as handle:
    textidx = pickle.load(handle)

#predicted factors from CB branch
#predicted_factors = np.load(os.path.join(files,"predicted_factors{}.npy".format(seed)))
#batch laoding predicted factors:

tmp = [[] for _ in range(35)]
for i in range(35):
    try:
        tmp[i] = np.load(os.path.join(files,"predicted_factors_{}_{}.npy".format(i,seed)))
        print("Ok, len = ",str(len(tmp[i])))
    except:
        print(str(i)," not found")
#batch injecting factors

injected_factors = item_factors.copy()
batchno = 0
for batch in tmp:
    if len(batch)>0:
        for i in range(len(batch)):
            injected_factors[textidx[batchno][i]] = batch[i]
    batchno+1


#Test users
with open (os.path.join(files,"testusers{}.pickle".format(seed)),'rb') as handle:
    test_users = pickle.load(handle) 

#popularity list
with open (os.path.join(files,"testpop{}.pickle".format(seed)),'rb') as handle:
    popularity = pickle.load(handle) 



random = np.random.uniform(low = -1,high = 1,size=(615142,100))

ran= eval(train_sparse,test_sparse,user_factors,random,test_users,popularity)

hh= eval(train_sparse,test_sparse,user_factors,injected_factors,test_users,popularity)

cf= eval(train_sparse,test_sparse,user_factors,item_factors,test_users,popularity)




