def load_dict(input_json_file):

    train = pd.read_json('/Users/pantera/melon/train.json')
    #keep only playlist id and songs information
    train2 = train[['id','songs']]
    #add number of songs per playlists
    train3 = train2.assign(song_counts = train2['songs'].apply(lambda x: len(x)))
    #discover playlist-song interactions. Saves the result in a dictionary
    def discover(playlist :list ,playid : int, catalogue :dict):
        for k in catalogue.keys():
            if k in playlist:
                catalogue[k].append(playid)
    train3.apply(lambda row: discover(row['songs'],row['id'],tracks_dict) , axis = 1)


