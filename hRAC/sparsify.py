def sparsify(cutoff,data  ,files ="/Users/pantera/melon/files"):
    """ Mask interactions in data accoring to popularity cutoff """
    """ Returns traindata ,a list of masked interactions and percentage drop interactions  """
    import pickle
    import os
    _return = dict()
    testdata = data.copy()
    traindata = list(range(len(testdata)))
    #get track popularity
    popularity = [len(x) for x in testdata]
    #masking cold start tracks (i.e. <= cutoff interactions)
    mask = [0 <= interactions <= cutoff for interactions in popularity]
    wipepop = [cutoff<interactions for interactions in popularity]
    #save results for later loading into audio module
    with open(os.path.join(files,"coldtracks.pickle"), 'wb') as handle:
        pickle.dump(mask, handle)
    with open(os.path.join(files,'poptracks.pickle'), 'wb') as handle:
        pickle.dump(wipepop, handle)
    masked = []
    #building train dataset
    for track in range(len(testdata)):
        if mask[track] == True:
            #keep a record of masked interactions. This way we know which users had their tracks masked in training.
            for playlist in testdata[track]:
                masked.append((playlist,track))
            traindata[track]=[]
        else:
            traindata[track] = testdata[track]
    with open(os.path.join(files,'masked_interactions.pickle'), 'wb') as handle:
        pickle.dump(masked, handle)
    #check density
    interactionstest = len([ (j,i) for j in range(len(testdata)) for i in range(len(testdata[j])) ])
    # 70,785,005,082 is total number of elements in user-item activity matrix
    sparsity = 1-len([ (j,i) for j in range(len(traindata)) for i in range(len(traindata[j])) ])/70785005082
    drop_pct = (len(masked)/interactionstest)*100
    _return['traindata'] = traindata
    _return['masked'] = masked
    _return['drop_pct'] = drop_pct
    _return['sparsity_pct'] = sparsity*100
    return _return
