from sklearn.cluster import MiniBatchKMeans

def train_codebook(params,descriptors,paraules):
    kMeans = MiniBatchKMeans(params['descriptor_size'])
    #Entrenem el KMeans amb els descriptors
    kMeans.fit(descriptors)
    return kMeans
    
