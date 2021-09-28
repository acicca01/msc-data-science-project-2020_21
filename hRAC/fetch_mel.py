def fetch_mel(trackset , root = '/Users/pantera/melon/arena_mel/',files = "/Users/pantera/melon/files"):
    """
    Checks Mel-Spectrograms are of the correct shape prior to loading

    Parameters
    ----------
    trackset = list of encoded trackids to be imported
    root = Mel_spectrograms file location
    files = project files location 

    Returns
    -------
    Returns a tuple (audio_files , retrieved)

    audio_files = list of paths to genuine mel-spectrograms
    retrieved = list of encoded ids of tracks in audio_files
    """

    import math
    import numpy as np
    import pickle
    import os 
    with open (os.path.join(files,"decodedmap_tracks.pickle"),'rb') as handle:
        decode = pickle.load(handle)

    decoded = [decode[x] for x in trackset]
    audio_files = []
    retrieved = []
    for i in range(len(decoded)):
        melpath = os.path.join(root,str(math.floor(decoded[i]/1000)),str(decoded[i])+".npy")
        with open(melpath,'rb') as melh:
            m,n = np.lib.format.read_magic(melh)
            shape , _ ,_ = np.lib.format.read_array_header_1_0(melh)
            if shape == (48,1876):
                audio_files.append(melpath)
                retrieved.append(trackset[i])
    return audio_files,retrieved
