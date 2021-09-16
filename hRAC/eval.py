def eval(train_sparse,test_sparse,user_factors,item_factors,userlist):
    """ Function used to evaluate recommendation """
    """ For each set of recommendation returns the mean AUC score """
    """ train --> training set in sparse format """
    """ test --> test_set in sparse format """
    """ user_factors , item_factors --> user , item factors """
    """ userlist --> list of users to run evalutation on """

    from sklearn import metrics
    import time
    import numpy as np
    from scipy.sparse import csr_matrix
    start = time.time()
    auc_scores = []
    tot = len(userlist)
    i = 0
    for user in userlist:
        training = train_sparse[user,:].toarray().reshape(-1)
        zeros = np.where(training == 0)[0]    #indixes for no interactions
        actual = test_sparse[user,:].toarray().reshape(-1)
        actual = actual[zeros]
        #subsetting predictions only on items in training with no interaction
        scores = user_factors[user,:].dot(item_factors.T)
        scores = scores[zeros]
        fpr, tpr, thresholds = metrics.roc_curve(actual, scores)
        auc_scores.append(metrics.auc(fpr, tpr))
        progress = i/tot
        print("\r>> Progress {:.0%}".format(progress),end='')
        i+=1
    print()
    print("generated scores in {} minutes ".format(int(time.time() -start)/60))
    return np.mean(auc_scores)
