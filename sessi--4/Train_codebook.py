from sklearn.cluster import MiniBatchKMeans

def train_codebook(n_clusters,descriptors,paraules):
    kMeans = MiniBatchKMeans(paraules)
    #Entrenem el KMeans, ajustem el numero de clusters a 4
    kMeans.fit(descriptores)
    return km
    
