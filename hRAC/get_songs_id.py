def get_songs_id(root="/Users/pantera/melon/") -> list:
    """Reads in all playlist informations from train.json,val.json and test.json and extract unique songs id for the 649,091 songs featured in the dataset"""
    train = pd.read_json(root+"train.json")
    test  = pd.read_json(root+"test.json")
    val   = pd.read_json(root+"val.json")
    
    songs = list(train['songs'])
    trainsongs = []
    for outerl in songs:
        for ele in outerl:
            trainsongs.append(ele)
    
    songs = list(test['songs'])
    testsongs = []
    for outerl in songs:
        for ele in outerl:
            testsongs.append(ele) 

    songs = list(val['songs'])
    valsongs = []
    for outerl in songs:
        for ele in outerl:
            valsongs.append(ele) 
    
    concat = list(set(trainsongs+testsongs+valsongs))
    return concat
