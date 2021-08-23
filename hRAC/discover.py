def discover(playlist :list ,playid : int, tracks_dict :dict):
    """ discover playlist-song interactions in a DataFrame. Saves the result in a dictionary """
    """ function to be applied to a pandas Dataframe where a playlist and playlistID """
    """ track_dict is a dictionary with keys the tracks in the DataFrame. This must be initialized before the call """

    for k in tracks_dict.keys():
        if k in playlist:
            tracks_dict[k].append(playid)


