import os
import numpy as np
import pickle
import math
from tensorflow.keras import optimizers
import tensorflow as tf
from hRAC.consolidate import consolidate
from hRAC.cnn import cnn
root = "/Users/pantera/melon/arena_mel_compress/"
files = "/Users/pantera/melon/files"

#gather data
with open('/Users/pantera/melon/files/coldtracks.pickle', 'rb') as handle:
    coldtracks = pickle.load(handle)
with open('/Users/pantera/melon/files/poptracks.pickle', 'rb') as handle:
    poptracks = pickle.load(handle)
with open('/Users/pantera/melon/files/decodemap', 'rb') as handle:
    decodemap = pickle.load(handle)

traintracks = [decodemap[i] for i in range(len(poptracks)) if poptracks[i] == True]
consolidate(traintracks,"training" , root)
trainset = np.load("/Users/pantera/melon/files/training.npy")
trainset = trainset.reshape(28393,48,48,1)
model = cnn(32)
adam = optimizers.Adam(lr=0.001)
model.compile(loss='MeanSquaredError', metrics='MeanAbsoluteError', optimizer=adam)
targetset = np.load("/Users/pantera/melon/files/track_factors.npy")
targetset = targetset[poptracks , :]
hh = model.fit(x=trainset,y=targetset,validation_split = 0.2,epochs = 20)

        



#add 1 channel for CNN
with open('/Users/pantera/melon/files/decodemap', 'rb') as handle:
    decode = pickle.load(handle)

#check training for first 100 compressed tracks
#                                                            #      ###      ###   
# #####  #####     ##    #  #    #  #  #    #   ####        ##     #   #    #   #  
#   #    #    #   #  #   #  ##   #  #  ##   #  #    #      # #    #     #  #     # 
#   #    #    #  #    #  #  # #  #  #  # #  #  #             #    #     #  #     # 
#   #    #####   ######  #  #  # #  #  #  # #  #  ###        #    #     #  #     # 
#   #    #   #   #    #  #  #   ##  #  #   ##  #    #        #     #   #    #   #  
#   #    #    #  #    #  #  #    #  #  #    #   ####       #####    ###      ###   
                                                                                  

