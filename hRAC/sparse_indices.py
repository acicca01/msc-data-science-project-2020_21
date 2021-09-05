def sparse_indices(interactions: list):
    """ Returns a triplet of parameters that can be used to fill in a sparse matrix (Playlist-Track interactions  utility matrix)"""

    data = []
    row_ind = []
    col_ind = []
    for i in range(len(interactions)):
        for playlist in interactions[i]:
            data.append(1)
            row_ind.append(playlist)
            col_ind.append(i)
    return data, row_ind, col_ind
