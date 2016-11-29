from sklearn.cluster import MiniBatchKMeans

def train_codebook(n_clusters,descriptors,paraules):
    kMeans = MiniBatchKMeans(paraules)
    #Entrenem el KMeans
    kMeans.fit(descriptores)
    return km
    
