# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing

def bow(assignments,kMeans):
    # Creem un descriptor de la mateixa mida que el número de clusters. Aquest descriptor està tot 
    #valor a cero
    descriptor = np.zeros(np.shape(kMeans.cluster_centers_)[0])
    # Per cada entrada a l'assigments, sumem 1 al índex que pertoca a l'histograma
    for a in assignments:
        descriptor[a] += 1
    # És important normalitzar amb L2 
    descriptor = preprocessing.normalize(descriptor)
    return descriptor