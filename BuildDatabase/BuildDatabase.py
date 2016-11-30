# -*- coding: utf-8 -*-

#importem la llibreria
import os 

#definim la funció build_database que necessitarà dos direccions per executar-la,
#la primera per llegir-ne les dades i la segona per escriure-hi els resultats.
def build_database (indir,outdir): 
    #guardem a la variable images una llista amb tots els noms dels fitxers presents en el directori indir.
    images = os.listdir(indir)  
    #creem en el directori d'escriptura un fitxer anomenat ID on escriurem els noms de les imatges llegides anteriorment.
    outfile = open(outdir+'/ID.txt','w')   
    for file in images: #recorre tots els id de la llista guardats a la variable imatges i els escriu al fitxer
        outfile.write(file[0:-4]+"\n") 
        print(file)
    outfile.close()#tanca el fitxer
    
#Creem les rutes absolutes:

#Direcció de lectura de les imatges de validació:
ruta1=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/val/images'
#Direcció de lectura de les imatges d'entrenament:
ruta2=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/train/images'
#Direcció d'escriptura de les imatges de validació:
savepath1=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/val'
#Direcció d'escriptura de les imatges d'entrenament:
savepath2=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/train'

#Executem la funció build_database definida anteriorment per a generar l'arxiu d'IDs per a les imatges de validació:
build_database(ruta1,savepath1)
#Executem la funció build_database definida anteriorment per a generar l'arxiu d'IDs per a les imatges d'entrenament:
build_database(ruta2,savepath2)
