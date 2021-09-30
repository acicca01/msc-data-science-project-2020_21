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
#importing seed from main session
with open (os.path.join(files,"seed.pickle"),'rb') as handle:
    seed = pickle.load(handle)

#loading validation set
valset = np.load(os.path.join(files,"valset" + str(seed)+".npy"))
valset = valset.reshape(valset.shape[0],valset.shape[1],valset.shape[2],1)
valfactors = np.load(os.path.join(files,"valset_factors"+ str(seed)+".npy"))


#loading training set
trainset = np.load(os.path.join(files,"trainset" + str(seed)+".npy"))
trainset = trainset.reshape(trainset.shape[0],trainset.shape[1],trainset.shape[2],1)
trainfactors = np.load(os.path.join(files,"trainset_factors"+ str(seed)+".npy"))


model = cnn(48,1876,100)
model.load_weights(os.path.join(files,"weights{}.hdf5".format(seed)))
adam = optimizers.Adam(lr=0.007)
model.compile(loss='MeanSquaredError', metrics='MeanAbsoluteError', optimizer=adam)


####TRAINING 
#earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
#                                        mode ="min", patience = 3,  
#                                        restore_best_weights = True)
#hh = model.fit(x=trainset,y=trainfactors,validation_data = (valset , valfactors), epochs = 50,callbacks = [earlystopping])
checkpoint = callbacks.ModelCheckpoint(os.path.join(files,"weights{}.hdf5".format(seed)),monitor='val_loss',verbose=1,save_best_only=True,mode='min')
hh = model.fit(x=trainset,y=trainfactors,validation_data = (valset , valfactors), epochs = 50, callbacks=[checkpoint])
plt.plot(hh.history['loss'], label='train')
plt.plot(hh.history['val_loss'], label='test')
plt.legend()
plt.figsave(os.path.join(files,"cnntraining{}".format(seed)))
#------------------------------------------
#               prediction

predicted = []
for i in range(35):
    testset = np.load(os.path.join(files,"testset_{}_{}.npy".format(i,seed)))
    testset = np.load(os.path.join(files,"testset" + str(seed)+".npy"))
    testset = testset.reshape(testset.shape[0],testset.shape[1],testset.shape[2],1)
    predicted.append(model.predict(testset))
    np.save(os.path.join(files,"predicted_factors{}.npy".format(seed)),predicted)

#alternative method will save multiple prediction files
#for i in range(35):
#    testset = np.load(os.path.join(files,"testset_{}_{}.npy".format(i,seed)))
#    testset = testset.reshape(testset.shape[0],testset.shape[1],testset.shape[2],1)
#    np.save(os.path.join(files,"predicted_factors_{}_{}.npy".format(i,seed)),model.predict(testset))


#------------------------------------------
#               evaluation 

results = []
for i in range(35):
    testset = np.load(os.path.join(files,"testset_{}_{}.npy".format(i,seed)))
    #testset_factors = np.load(os.path.join(files,"testset_factors{}.npy".format(seed)))
    testset_factors = np.load(os.path.join(files,"testset_factors_{}_{}.npy".format(i,seed)))
    results.append(model.evaluate(testset,testset_factors))




