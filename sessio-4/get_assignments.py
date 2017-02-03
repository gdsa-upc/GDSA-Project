def get_assignments (KMeans, descriptors):
    
    #Calculem les assignacions mitjancant la funcio predict( )
    assignments=KMeans.predict(descriptors)
    #Retornem el vector amb les assignacions per cada descriptor
    return assignments