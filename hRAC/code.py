def code(idlist , files= "/Users/pantera/melon/files/"):
    """ Encode or Decode Playlists/Track ids to a contiguous sequence
        It saves the encodemap and decodemap in files                """
    #idlist_to_sequence
    encodemap = {y :x  for (x,y) in enumerate(idlist)}
    with open(os.path.join(files,"encodemap"), 'wb') as handle:
        pickle.dump(encodemap, handle)
    #sequence_to_idlist
    decodemap = {x :y  for (x,y) in enumerate(idlist)}
    with open(os.path.join(files,"decodemap"), 'wb') as handle:
        pickle.dump(decodemap, handle)
    return (encodemap,decodemap)
