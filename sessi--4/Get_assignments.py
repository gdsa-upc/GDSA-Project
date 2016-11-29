#Amb aquesta funció cada descriptor local (obtinguts anteriorment) s'assigna al cluster que li correspon (per a fer-ho es mira quin
#és el centroid més pròxim).  Cal recordar que descriptors és una matriu amb forma=(n_samples,n_features)
def get_assignments (KMeans, descriptors): #kmeans és la sortida del codebook
    
    #Amb la funció predict podem calcular l'index del cluster per a cada mostra.
    assignments=KMeans.predict(descriptors)
    #Retornem el vector amb els index assignats per cada descriptor
    return assignments
