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
train_sparse = sparse.load(os.path.join(files,"train_sparse_{}.npz".format(seed)))
test_sparse = sparse.load(os.path.join(files,"test_sparse_{}.npz".format(seed)))

item_factors = np.load(os.path.join(files,"item_factors_{}".format(seed)))
user_factors = np.load(os.path.join(files,"user_factors_{}".format(seed)))

with open (os.path.join(files,"testidx{}".format(seed)),'rb') as handle:
    textidx = pickle.load(handle)

predicted_factors = np.load(os.path.join(files,"predicted_factors{}".format(seed)))

