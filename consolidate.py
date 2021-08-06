def consolidate(songs: list,allin: str,root= "/Users/pantera/melon/arena_mel_compress/"):
    """Recursively look for songs data in root and consolidate them into a single file allin. The consolidated file is saved in root"""
    bigone = []
    for song in songs:
        loadthis = os.path.join(root,str(math.floor(song/1000)),str(song)+".npy")
        bigone.append(np.load(loadthis))
    bigonenp = np.array(bigone)
    np.save(root+allin,bigonenp)    
