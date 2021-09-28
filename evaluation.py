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
predicted_factors = np.load(os.path.join(files,"predicted_factors{}.npy".format(seed)))

#Test users
with open (os.path.join(files,"testusers{}.pickle".format(seed)),'rb') as handle:
    test_users = pickle.load(handle) 

#popularity list
with open (os.path.join(files,"testpop{}.pickle".format(seed)),'rb') as handle:
    popularity = pickle.load(handle) 



# check auc for collaborative filtering branch
#mAUC= eval(train_sparse,test_sparse,user_factors,item_factors,test_users,AUC)
(auc,ndcg,mapk) = eval(train_sparse,test_sparse,user_factors,item_factors,test_users,NDCG)




injected_factors = item_factors.copy()
i = 0
for ele in textidx:
    injected_factors[ele] = predicted_factors[i]
    i +=1
injected_factors == item_factors
