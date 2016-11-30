# -*- coding: utf-8 -*-
## Ranking
import os
import cPickle as pk
from sklearn import metrics

def rank(params):
    #Obrim els fitxers de característiques de validació i train
    fval= open((os.path.join(params['root'],params['database'],'val','Features.txt')),'r')
    ftrain= open((os.path.join(params['root'],params['database'],'train','Features.txt')),'r')
    #Obrim el fitxer de les annotation
    annotation=open((os.path.join(params['root'],params['database'],'val','annotation.txt')),'r')
    #Aquest readline() és per saltar-nos els titols "ClassID i ImageID" 
    annotation.readline()    
    #Declarem la llista que portarà les ID de la classe desconegut per poder descartar-les
    lista_desconegut=[]
    for line in annotation:
        ImageID=line.split()[0]
        ClassID=line.split()[1]
        #Segons el .txt entrant, el primer paràmetre que trobem a cada linia és la ImageID i el segon la ClassID
        if ClassID=='desconegut':
            lista_desconegut.append(ImageID)
    #Si la ClasseID és desconegut, posem la ImageID a la llista
    #Declarem els dictionaries
    dval=dict()
    dtrain=dict()
    #Carreguem la informació als dictionaries
    dval=pk.load(fval)
    dtrain=pk.load(ftrain)
    #Generem una llista amb les keys de train
    lhtrain=list()
    lktrain=dtrain.keys()
    #Generem una llista amb els histogrames del diccionari train
    for tkey in lktrain:
        lhtrain.append(dtrain[tkey])
        
    #Declaració de variables 
    ldist=list()#Llista de distàncies entre l'histograma de validació i els de train
    tup=()#Tupla on guardarem la ID de train i la distància al histograma de validació 
    ltup=list()#Llista de tuples 
    #Per cada imatge de Validació generem un ranking 
    for vkey in dval:
        #No evaluem la classe desconegut
        if vkey not in lista_desconegut:
            #Obrim el fitxer on escriurem el ranking per la imatge de validació
            frank = open((os.path.join(params['root'],params['database'],'val','result',vkey+'.txt')),'w')
            #Calculem la distància entre l'histograma de validació i els de train
            ldist=metrics.pairwise.pairwise_distances(dval[vkey],lhtrain,metric='euclidean',n_jobs=1)[0]
            #Creem una llista de tuples amb les IDs i les distancies de train
            ldist=list(ldist)
            for tkey in lktrain:
                tup=(tkey,ldist.pop(0))
                ltup.append(tup)
            #Ordenem la llista segons la distancia 
            ltup=sorted(ltup, key=lambda distancia: distancia[1])
            #Escrivim cada ID ordenada al fitxer rank
            for tupla in ltup:
                frank.write(tupla[0]+"\n")
            frank.close()
            ltup=[]
    #Tanquem els fitxers de característiques de validació i train
    fval.close()
    ftrain.close()