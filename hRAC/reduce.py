def reduce(components:int,source="/Users/pantera/melon/arena_mel"):
    """Dimensionality reduction with PCA, from 1876 down to components"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pca = PCA(n_components=components)
    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            source = os.path.join(root, name)
            destination = source.replace("arena_mel","arena_mel_compress")
            try:
                tmp = np.load(source)
                tmp = tmp.astype(np.double)
                tmp_scaled = scaler.fit_transform(tmp)
                tmp2 = pca.fit_transform(tmp_scaled)
            except:
                print(name, "failed")
            try:
                np.save(destination, tmp2, allow_pickle=True, fix_imports=True)
            except: 
                ddir = destination[:(len(destination)-len(name)-1)]
                os.mkdir(ddir)
                np.save(destination, tmp2, allow_pickle=True, fix_imports=True)

