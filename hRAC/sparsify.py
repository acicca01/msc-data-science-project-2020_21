def sparsify(testdata,pct,seed):
    """
    Drops interactions from a dataset.
    
    Parameters
    ----------
    testdata : data for original interactions passed as a list of lists     
    pct : percentage of interactions to be dropped
    seed : random seed

    Returns
    -------
    It returns a dictionary with following key-value pairs

    trainingdata : sparsified dataset
    userdropped :  list of users affected from the dropping
    trackdropped : list of dropped tracks 
    
    Notes
    -----
    (testdata[i] = list of playlistIDs interacting with trackID i)
    """
    import numpy as np
    np.random.seed(seed)
    _return = dict()
    #random mask of interactions
    interactionset = [ (user,item) for item in range(len(testdata)) for user in testdata[item] ]
    randomindices = np.random.choice(len(interactionset),int(len(interactionset)*((100-pct)/100)) , replace = False)
    interactionmask = []
    #for i in range(len(randomindices)):
    #    interactionmask.append(interactionset[i])
    #trainingdata = []
    for idx in randomindices:
        interactionmask.append(interactionset[idx])
    trainingdata = []    
    for i in range(len(testdata)):
        trainingdata.append([])
    for ele in interactionmask:
        trainingdata[ele[1]].append(ele[0])
    A = set(interactionset)
    B = set(interactionmask)
    dropped = list(A-B)
    userdropped = list(set([x[0] for x in dropped]))
    trackdropped = list(set([x[1] for x in dropped]))
    _return['traindata']=trainingdata
    _return['userdropped']=userdropped
    _return['trackdropped']=trackdropped
    return _return 
    
