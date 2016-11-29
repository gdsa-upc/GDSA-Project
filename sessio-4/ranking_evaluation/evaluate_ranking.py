# -*- coding: utf-8 -*-
import os 
from get_params import get_params

def evaluate_ranking():
    params=get_params()
    f=open((params['root']+'/'+params['database']+'/train/annotation.txt'),'r') #obrim el fitxer ground truth del train i llegim l'interior
    next(f)
    dic_gtt={} #crea un diccionari amb la ground truth de train
    for line in f:
        a=line.index('\t')
        dic_gtt[line[0:a]]=line[a+1:]
    f.close()        
    f=open((params['root']+'/'+params['database']+'/'+params['val']+'/annotation.txt'),'r') #fem el mateix amb els del test o validació
    next(f)
    dic_gtv={} #crea un diccionari amb la ground truth de validació o test
    for line in f:
        a=line.index('\t')
        dic_gtv[line[0:a]]=line[a+1:]
    f.close() 
    ranks = os.listdir(params['root']+'/rankings/')  #llegeixo tots els noms dels fitxers del directori d'entrada i els guarda a ranks
    sumap=0 #suma dels average precisions
    contr=0
    lista=[]
    for r in ranks:
        #llegeix els resultats de cada rank
        R=r.rstrip('.txt')
        if dic_gtv[R]!='desconegut':
            contr=contr+1#contador de ranks
            c=dic_gtv[R]
            f=open(params['root']+'/rankings/'+r)
            lines=f.read().split()
            f.close()
            cont=0 #comptador de resultats
            ce=0 #comptador de resultats encertats
            sump=0 #suma de precisions
            for l in lines:
                if l!='Thumb':
                    cont=cont+1
                    if dic_gtt[l]==c:
                        ce=ce+1
                        sump=sump+(float(ce)/float(cont))
            ap=float(sump)/float(ce)
            lista.append(ap)
            sumap=float(sumap)+float(ap)
    print 'mean average precision='
    print'\n'
    return (float(sumap)/float(contr))  