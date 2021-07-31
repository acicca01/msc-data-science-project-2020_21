def reduce(source="/Users/pantera/melon/arena_mel"):
    i = 0
    for root, dirs, files in os.walk(source, topdown=False):
        for name in files:
            i +=1
            if i <10:
                source = os.path.join(root, name)
                destination = source.replace("arena_mel","arena_mel_compress")
                tmp = np.load(source)
                pca = PCA(n_components=10)
                pca.fit(tmp)
                tmp2 = pca.transform(tmp)
                try:
                    np.save(destination, tmp2, allow_pickle=True, fix_imports=True)
                except: 
                    ddir = destination[:(len(destination)-len(name)-1)]
                    os.mkdir(ddir)
                    np.save(destination, tmp2, allow_pickle=True, fix_imports=True)
            else:
                break
        
