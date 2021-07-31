# Create dummy music collection
# Music collection will consist of 10 playlist featuring 10 songs each. Total song in collection is 14

def dummify(play_no,songs_in_cat,songs_in_playlists):
# play_no --> number of playlists
# songs_in_cat --> total number of songs in catalogue
# songs_in_playlists --> number of songs in playlist
    import pandas as pd
    import numpy as np

    # create playlist identifiers
    plylst_id = np.arange(0,play_no,1)

    # create 14 random weights 
    weights = np.random.choice(4*songs_in_cat,songs_in_cat)
    somma = sum(weights)

    # probs is a list of probabilities, with prob[i] = probability i-th song of the collection is selected for a playlist 
    probs=[]
    for ele in weights:
        probs.append(ele/somma)

    mysongs=[]
    for i in range(10):
        mysongs.append(np.random.choice(songs_in_cat,songs_in_playlists, p = probs, replace = False))

    dummy = pd.DataFrame()
    dummy["plylst_id"] = pd.Series(plylst_id)
    dummy["songs"] = pd.Series(mysongs)
    return dummy






