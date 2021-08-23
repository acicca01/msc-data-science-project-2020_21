def reduce(components:int,target: str,source="/Users/pantera/melon/arena_mel"):
    """Dimensionality reduction with PCA, from 1876 down to components"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pca = PCA(n_components=components)
    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            #source file to compress
            source = os.path.join(root, name)
            #destination directory 
            destdir =os.path.join(target,source.split('/')[-2])
            #full path to target file
            target_file =  os.path.join(destdir,name)
            try:
                tmp = np.load(source)
                tmp = tmp.astype(np.double)
                tmp_scaled = scaler.fit_transform(tmp)
                tmp2 = pca.fit_transform(tmp_scaled)
            except:
                print(name, "failed")
            try:
                np.save(target_file, tmp2, allow_pickle=True, fix_imports=True)
            except: 
                os.mkdir(destdir)
                np.save(target_file, tmp2,allow_pickle=True, fix_imports=True)

