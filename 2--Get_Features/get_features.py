# -*- coding: cp1252 -*-
import os, sys
import numpy as np
import pickle as pk
from local_feature_extraction import local_feature_extraction
from train_codebook import train_codebook
from get_assignments import get_assignments
from build_bow import build_bow
from get_params import get_params



def get_features(params):
    #-------- Imatges d'entrenament --------#

    #Obrim el fitxer que conte les ID de les imatges d'entrenament
    ID=open(os.path.join(params['root'],params['database'],'train','ID.txt'), 'r')
    #Extraccio de les caracteristiques de la imatge de la primera linia del ID.txt, la funcio readline() llegeix
    #una sola linea d'un fitxer, aquesta lectura llegeix un caràcter "\n" al final de la linea, menys quan arriva al
    #final del fitxer (l'absencia d'aquest indica el final) però com que no ens interessa tenir el caràcter al final
    #del nom, emperem la subfunció .replace() que substituirà aquest caràcter per un espai en blanc ' '.
    nom=str(ID.readline()).replace('\n','')
    #invoquem la funciṕ local_feature_extraction per la primera imatge que hem llegit guardada a "nom".
    #Amb os.path.join() obtenim la imatge del directori.
    if os.path.exists(os.path.join(params['root'],params['database'],'train','images', nom + '.jpg')):
        image=os.path.join(params['root'],params['database'],'train','images', nom + '.jpg')
    if os.path.exists(os.path.join(params['root'],params['database'],'train','images', nom + '.JPG')):
        image=os.path.join(params['root'],params['database'],'train','images', nom + '.JPG')

    des_train=local_feature_extraction(params,image)
    #Creem un diccionary amb la funció dic() on i guardarem els descriptors per a cada imatge.
    # A diferència de les seqüències que estan indexades per un rang de numeros,els diccionaris són com matrius que estàn
    #indexats per keys les quals poden ser de qualsevol tipus immutable: strings i numeros poden ser sempre keys.
    dic_train=dict()
    #Guardem per a la primera imatge amb ID "nom", els seus descriptors associats en el diccionari per a imatges d'entrenament:
    dic_train[nom]=des_train
    #Generem un bucle per fer el mateix amb la resta d'imatges d'entrenament.
    for line in ID:
        nom=str(line).replace('\n','')

        if os.path.exists(os.path.join(params['root'],params['database'],'train','images', nom + '.jpg')):
            image=os.path.join(params['root'],params['database'],'train','images', nom + '.jpg')
        if os.path.exists(os.path.join(params['root'],params['database'],'train','images', nom + '.JPG')):
            image=os.path.join(params['root'],params['database'],'train','images', nom + '.JPG')

        x=local_feature_extraction(params,image)
        #Creem un vector que contindrà tots els descriptors
        des_train=np.concatenate((des_train,x))
        #Per a cada imatge tindrem a la matriu els seus descriptors.
        dic_train[nom]=x
    #Tanquem el fitxer
    ID.close()

    #Calculem els centroids "entrenant" la funció KMeans amb el numero de clusters(paraules) que volem i amb els descriptors
    #de les imatges d'entrenament.
    clusters=100
    codebook=train_codebook(params,des_train,clusters) #des_train conté tots els descriptors de cada imatge concatenats
    #Obrim el fitxer que conte les ID de les imatges d'entrenament per poder llegirlo altre cop des de l'inici:
    ID=open(os.path.join(params['root'],params['database'],'train','ID.txt'), 'r')

    for line in ID:
        nom=str(line).replace('\n','')
        #Calculem els index de cluster per cada mostra. Utilitzem el diccionari creat anteriorment per obtenir els descriptors
        #per a cada imatge i els centroids definits amb la funció train_codebook:
        assignments=get_assignments(codebook,dic_train[nom])
        #Substituim per a cada imatge l'assignació que tenia en el diccionari per un histograma(Bow) que indicarà quants descriptors
        #hi ha per cada paraula(cluster):
        dic_train[nom]=build_bow(assignments,codebook)
    #Tanquem el fitxer
    ID.close()

    #Guardem el diccionari amb el BoW per cada imatge d'entrenament en l'arxiu "Features.txt".
    bow_train = open(os.path.join(params['root'],params['database'],'train','Features.p'), 'wb')
    #la funció dump() permet guardar la variable dic_train en l'arxiu features.txt indicat per la variable bow_train.
    pk.dump(dic_train,bow_train)
    bow_train.close() #Tanquem el fitxer features.txt
    feat=open(os.path.join(params['root'],params['database'],'train','Features.p'), 'rb')
    p = pk.load(feat)
    feat.close()

    #---------Imatges de validació ---------#

    #Fem els mateixos passos anteriors però per les imatges de validació:
    #Obrim el fitxer que conté les ID de les imatges de validacio
    ID = open(os.path.join(params['root'],params['database'],'val','ID.txt'), 'r')
    nom=str(ID.readline()).replace('\n','')

    if os.path.exists(os.path.join(params['root'],params['database'],'val','images', nom + '.jpg')):
        image=os.path.join(params['root'],params['database'],'val','images', nom + '.jpg')
    if os.path.exists(os.path.join(params['root'],params['database'],'val','images', nom + '.JPG')):
        image=os.path.join(params['root'],params['database'],'val','images', nom + '.JPG')

    des_val=local_feature_extraction(params,image)
    #Creacio del diccionari de les imatges de validacio
    dic_val=dict()
    dic_val[nom]=des_val
    for line in ID:
        nom=str(line).replace('\n','')
        if os.path.exists(os.path.join(params['root'],params['database'],'val','images', nom + '.jpg')):
            image=os.path.join(params['root'],params['database'],'val','images', nom + '.jpg')
        if os.path.exists(os.path.join(params['root'],params['database'],'val','images', nom + '.JPG')):
            image=os.path.join(params['root'],params['database'],'val','images', nom + '.JPG')
        #Extraccio de les caracteristiques (keypoints) per a les imatges de validacio
        x=local_feature_extraction(params,image)
        des_val=np.concatenate((des_val,x)) #Creem un vector que contè tots els descriptors
        #Omplim el diccionari amb els descriptors per cada imatge.
        dic_val[nom]=x
    #Tanquem el fitxer
    ID.close()

    #Amb el mateix numero de clusters definits anteriorment i amb el nou vector de descriptors per les imatges de validació entrenem
    #la funció KMeans per a que creï els centroids.
    codebook=train_codebook(params,des_val,clusters)
    #Obrim el fitxer que conte les ID de les imatges de validació per poder llegirlo altre cop des de l'inici:
    ID=open(os.path.join(params['root'],params['database'],'val','ID.txt'), 'r')
    for line in ID:
        nom=str(line).replace('\n','')
        #Calculem els index de cluster per cada mostra. Utilitzem el diccionari creat anteriorment per obtenir els descriptors
        #per a cada imatge i els centroids definits amb la funció train_codebook:
        assignments=get_assignments(codebook,dic_val[nom])
        #Substituim per a cada imatge l'assignació que tenia en el diccionari per un histograma(Bow) que indicarà quants descriptors
        #hi ha per cada paraula(cluster):
        dic_val[nom]=build_bow(assignments,codebook)
    #Tanquem el fitxer
    ID.close()

    #Guardem el diccionari amb el BoW de les imatges de validació en l'arxiu "Features.txt"
    bow_val = open(os.path.join(params['root'],params['database'],'val','Features.p'), 'wb')
    pk.dump(dic_val,bow_val)
    bow_val.close()

    feat=open(os.path.join(params['root'],params['database'],'val','Features.p'), 'rb')
    p = pk.load(feat)
    feat.close()
    print (p)

params=get_params()
get_features(params)
