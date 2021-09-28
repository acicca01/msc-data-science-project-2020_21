def eval(train_sparse,test_sparse,user_factors,item_factors,userlist,popularity):
    """ Function used to evaluate recommendation """
    """ For each set of recommendation returns the mean AUC score """
    """ train --> training set in sparse format """
    """ test --> test_set in sparse format """
    """ user_factors , item_factors --> user , item factors """
    """ userlist --> list of users to run evalutation on """
    """ popularity --> list of popularity of all tracks in original catalogue (i.e. TEST data)  """ 

    from sklearn import metrics
    import time
    import numpy as np
    from scipy.sparse import csr_matrix
    from hRAC.metrics import ndcg,dcg,apk,mapk
    start = time.time()
    auc_scores = []
    ndcg_scores = []
    map_scores = []
    pop_score = []
    tot = len(userlist)
    i = 0
    for user in userlist:
        #auc score 
        training = train_sparse[user,:].toarray().reshape(-1)
        zeros = np.where(training == 0)[0]    #indixes for no interactions
        actual = test_sparse[user,:].toarray().reshape(-1)
        actualz = actual[zeros]
        #subsetting predictions only on items in training with no interaction
        scores = user_factors[user,:].dot(item_factors.T)
        scoresz = scores[zeros]
        fpr, tpr, thresholds = metrics.roc_curve(actualz, scoresz)
        auc_scores.append(metrics.auc(fpr, tpr))
        
        gtpos = np.where(actual == 1)[0]
        gtpos_train = np.where(training == 1)[0]
        #ground truth items
        gt = [x for x in gtpos if x not in gtpos_train]
        scoresrank = [(i,k) for i,k in enumerate(scores)]
        scoresrank.sort(key=lambda x :x[1],reverse = True) 
        #recommended items
        reco = [x[0] for x in scoresrank]
        #ndcg scores
        ndcg_scores.append(ndcg(gt,reco,len(reco)+1))
        #apk scores
        map_scores.append(apk(gt,reco,len(reco)+1))
        progress = i/tot
        #pop score
        popuser = 0
        for ele in reco[:100]:
            popuser+=popularity[ele]
        popuser = popuser/100
        pop_score.append(popuser)
        print("\r>> Progress {:.0%}".format(progress),end='')
        i+=1
    print()
    print("generated scores in {} minutes ".format(int(time.time() -start)/60))
    return np.mean(auc_scores) , np.mean(ndcg_scores) , np.mean(map_scores),np.mean(pop_score)

