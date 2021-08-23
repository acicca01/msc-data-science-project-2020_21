def code(idlist , mode = "EN"):
    """ Encode or Decode Playlists/Track ids to a contiguous sequence """ 
    if mode == 'EN':
        #idlist_to_sequence 
        return {y :x  for (x,y) in enumerate(idlist)}  
    if mode == "DE":    
        #sequence_to_idlist 
        return {x :y  for (x,y) in enumerate(idlist)}
