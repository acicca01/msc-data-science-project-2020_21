import os
import numpy as np
import pickle
import math
from tensorflow.keras import optimizers
import tensorflow as tf
from hRAC.cnn import cnn
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

files = "/Users/pantera/melon/files"


##Set seed before training
seed = 5132700
traincnn_factors = np.load(os.path.join(files,"traincnn_factors_"+ str(seed)+".npy"))
trainset = np.load(os.path.join(files,"trainset" + str(seed)+".npy"))
trainset = trainset.reshape(trainset.shape[0],trainset.shape[1],trainset.shape[2],1)
model = cnn(48,1876,100)
adam = optimizers.Adam(lr=0.01)
model.compile(loss='MeanSquaredError', metrics='MeanAbsoluteError', optimizer=adam)


####TRAINING 
#earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
#                                        mode ="min", patience = 5,  
#                                        restore_best_weights = True)
#hh = model.fit(x=trainset,y=traincnn_factors,validation_split = 0.2,epochs = 100,callbacks = [earlystopping])
hh = model.fit(x=trainset,y=traincnn_factors,validation_split = 0.2,epochs = 50)
plt.plot(hh.history['loss'], label='train')
plt.plot(hh.history['val_loss'], label='test')
plt.legend()
plt.show()

#------------------------------------------
#               prediction

#loading paths to testfiles
with open(os.path.join(files,"test_files{}".format(seed)), 'rb') as handle:
    test_files = pickle.load(handle)


good = []
bad = []

#filter out incomplete audio samples
#append path for good files in good
def audiocheck(audiolist):
    for x in audiolist:
        mel = np.load(x)
        if mel.shape == (48,1876):
            good.append(x)
        else:
            bad.append(x)

audiocheck(test_files)

def predict(x):
    mel = np.load(x)
    mel=mel.reshape(1,48,1876,1)
    y = model.predict(mel)
    return y

tmp = map(predict,good)
predicted = np.stack(list(tmp))
with open(os.path.join(files,"predicted_track_factors_{}".format(seed)), 'wb') as handle:
    pickle.dump(predicted, handle)



