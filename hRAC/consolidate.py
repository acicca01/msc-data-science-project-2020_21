def consolidate(songs: list,allin: str,root= "/Users/pantera/melon/arena_mel/",files = "/Users/pantera/melon/files/" ):
    """Recursively look for songs data in root and consolidate them into a single file allin. The consolidated file is saved in root"""
    import os
    import math
    import numpy as np
    bigone = []
    for song in songs:
        loadthis = os.path.join(root,str(math.floor(song/1000)),str(song)+".npy")
        bigone.append(np.load(loadthis))
    try:    
        bigonenp = np.asarray(bigone)
        np.save(files+allin,bigonenp)
    except:
        print(song , "Failed")
