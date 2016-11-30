# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing

#assignments és la sortida de la funció get_assignments que assigna un index de cluster a cada descriptor.
def build_bow(assignments,kMeans): #kmeans és la sortida de la funció train_codebook (assigna el centroid a un descriptor)
    # Definim el vector "l'histograma" bag of words per a que tingui la mateixa mida que el numero de clusters que tinguem.
    bow = np.zeros(np.shape(kMeans.cluster_centers_)[0]) #cluster_centers_ és una matriu [n_clusters, n_features]
    # Assigments retorna un vector amb l'idex de cluster assignat a cada descriptor.
    # Construim un histograma sumant +1 per a cada descriptor assignat a un index de cluster.
    for a in assignments:
        bow[a] += 1
    # És important normalitzar amb L2
    bow = preprocessing.normalize(bow)
    return bow
