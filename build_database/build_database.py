# -*- coding: utf-8 -*-
# importar imatges: 

import os
#os.chdir('/Users/Marta/Desktop/fotos escriptori')


#Definim la funció build_database
def build_database(params):
    #Declarem la llista d'ImageIDs
    ImageIDs = []
    #Retorna una llista dels arxius al path introduit
    lista = os.walk(os.path.join(params['root'],params['database'],params['split'],'images')) 
    #Bucle per omplir la llista amb els nombres ID de les imatges sense la
    #extensió
    for root, dirs, files in lista:
        for fichero in files:
            (nombreFichero, extension) = os.path.splitext(fichero)
            ImageIDs.append(nombreFichero)
    #Guardem la longitud de ImageIDs en la variable i
    i=len(ImageIDs)
    #Creem el .txt que contindrà IDs de les imatges
    archi=open(os.path.join(params['root'],params['database'],params['split'],'ImageIDs.txt'),'w')
    #Bucle per omplir el .txt amb les IDs de les imatges
    for i in list(range(i)):
        if(i<len(ImageIDs)-1):
            archi.write(ImageIDs[i] + '\n')
        else:
            archi.write(ImageIDs[i])
    #Tanquem l'arxiu .txt 
    archi.close()
    print("hola")