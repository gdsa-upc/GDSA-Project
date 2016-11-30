# -*- coding: utf-8 -*-
#funció que codifica els descriptors obtinguts en la local_feature_extraction i els assigna el centroid més pròxim
#que determinarà el cluster al que es troba el descriptor en la funció get_assignments.
from sklearn.cluster import MiniBatchKMeans

def train_codebook(params,descriptors,clusters):
    kMeans = MiniBatchKMeans(clusters)

    #funció que calcula els centroids agrupant-los en mini lots. Defineix els centroids amb un entrenament segons els descriptors
    #i el numero de clusters (paraules) que volem.
    kMeans.fit(descriptors)
    return kMeans
