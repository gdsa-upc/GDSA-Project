# -*- coding: utf-8 -*-
import os 

def build_database (indir,outdir): #definim la funció
    images = os.listdir(indir)  #llegeixo tots els noms dels fitxers del directori d'entrada i els guarda a imatges
    outfile = open(outdir+'/ID.txt','w')   #crea el fitxer on escriu els id.
    for file in images: #recorre tots els id i els escriu al fitxer
        outfile.write(file[0:-4]+"\n") 
        print(file)
    outfile.close()#tanca el fitxer
    

ruta1=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/val/images'
ruta2=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/train/images'
savepath1=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/val'
savepath2=os.path.dirname(os.path.abspath(__file__))+'/TerrassaBuildings900/train'

build_database(ruta1,savepath1)
build_database(ruta2,savepath2)
