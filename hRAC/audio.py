import os
import numpy as np
import pickle
import math
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.activations import elu
from tensorflow.keras.layers import BatchNormalization
from hRAC.consolidate import consolidate
root = "/Users/pantera/melon/arena_mel_compress/"
files = "/Users/pantera/melon/files"
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
                                                                                  
consolidate([x for x in range(100)],"train_100",root)
train = np.load(os.path.join(files,"train_100.npy"))
train = train.reshape(100,48,48,1)       
#determining right shape for input data

def cnn(melgram_input,n_factors):
    # Input block
    x = BatchNormalization(axis=1, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(64, 3, 3, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=3,  name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, padding='same', name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, padding='same', name='conv3')(x)
    x = BatchNormalization(axis=3,  name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_sie=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, padding='same', name='conv4')(x)
    x = BatchNormalization(axis=3,  name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)

    # Conv block 5
    x = Convolution2D(64, 3, 3, padding='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis,  name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    #output 
    x = Flatten()(x)
    x = Dense(n_factors,activation='linear',name='output')(x)

    # Create model
    model = Model(melgram_input,x)
    return model
