def sparsify(cutoff,data = testdata,files ="/Users/pantera/melon/files"):
    """ Mask interactions in data accoring to popularity cutoff """
    """ Returns traindata ,a list of masked interactions and percentage drop interactions  """
    import pickle
    import os
    traindata = list(range(len(alltracks)))
    #get track popularity
    popularity = [len(x) for x in testdata]
    #masking cold start tracks (i.e. <= cutoff interactions)
    mask = [0 <= interactions <= cutoff for interactions in popularity]
    wipepop = [cutoff<interactions for interactions in popularity]
    #save results for later loading into audio module
    with open(os.path.join(files,"coldtracks.pickle"), 'wb') as handle:
        pickle.dump(mask, handle)
    with open(os.pth.join(files,'poptracks.pickle'), 'wb') as handle:
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
    drop_pct = (len(masked)/interactionstest)*100
    return (traindata,masked,drop_pct)a
