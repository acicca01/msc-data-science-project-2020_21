def recommend(play  ,plays_factors, tracks_factors  ):
    """Builds a list of recommended tracks for a given playlist.Tracks are order by their score (descending order)

    play = playlist ID (encoded)
    plays_factors = playlists factors from the MF 
    tracks_factors = tracks factors from the MF / CNN 

    """
    score = plays_factors[play,:].dot(tracks_factors.T)
    recolist = [ (track,score[track]) for track in range(len(score))]
    recolist.sort(key=lambda x:x[1],reverse=True)
    return recolist
