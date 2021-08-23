def sparse_indices(interactions: dict):
    """ Returns a triplet of parameters that can be used to fill in a sparse matrix (Playlist-Track interactions  utility matrix)"""

    data = []
    row_ind = []
    col_ind = []
    for k,v in interactions.items():
        for playlist in v:
            data.append(1)
            row_ind.append(playlist)
            col_ind.append(k)
    return data, row_ind, col_ind
