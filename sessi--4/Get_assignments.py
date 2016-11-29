#Amb aquesta funció cada descriptor local (obtinguts anteriorment) s'assigna al cluster que li correspon (per a fer-ho es mira quin
#és el centroid més pròxim).
def get_assignments (KMeans, descriptors):
    
    #Calculem les assignacions mitjancant la funcio predict( )
    assignments=KMeans.predict(descriptors)
    #Retornem el vector amb les assignacions per cada descriptor
    return assignments
