# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def local_feature_extraction(params,image):
    #llegim la imatge:
    img = cv2.imread(os.path.join(params['root'],params['database'],'train','images',image + '.jpg'))
    #Cambiem la mida de la imatge:
    if not img is None:
        img1 = resize_image(params,img)
        #linea que soluciona un bug de opencv
        cv2.ocl.setUseOpenCL(False)

        # Creem l'objecte ORB que tindrà 200k keypoints. (Perametre que podem modificar per no saturar el programa)
        orb = cv2.ORB_create(2000)

        # Detectem els keypoints:
        kp = orb.detect(img1,None)

        # Calculem els descriptors amb els keypoints trobats.
        kp, des = orb.compute(img1, kp)

        # la sortida de la funció serà els descriptors
        return des

    else:
        img = cv2.imread(os.path.join(params['root'],params['database'],'train','images',image + '.JPG'))
        img1 = resize_image(params,img)
        #linea que soluciona un bug de opencv
        cv2.ocl.setUseOpenCL(False)

        # Creem l'objecte ORB que tindrà 200k keypoints. (Perametre que podem modificar per no saturar el programa)
        orb = cv2.ORB_create(2000)

        # Detectem els keypoints:
        kp = orb.detect(img1,None)

        # Calculem els descriptors amb els keypoints trobats.
        kp, des = orb.compute(img1, kp)

        # la sortida de la funció serà els descriptors
        return des


#Definim la funció resize_image per cambiar la mida de les imatges massa grans mantenint
#les proporcions (aspect ratio) height/width.
def resize_image(params,img):
    #Obtenim les dimensions de l'imatge amb la funció de numpy, shape.
    height, width = img.shape[:2]

    # En cas que la mida de la imatge sigui més petita que el parametre max_size
    #(mida maxima que especifiquem al invocar la funció), manté la mida width original.
    #Si la mida width de la imatge és més gran, passa a ser de la mida max_size.
    resize_dim = min(params['max_size'],width)
    #Per no perdre la relació d'aspecte normalitzem ara l'altura height segons la
    #relació entre l'amplada obtinguda i l'amplada original, per evitar numeros decimals arrodonim amb ceil
    dim = (resize_dim, math.ceil(height * resize_dim/width))

    img2=cv2.resize(img,dim)
    # La funció retorna una imatge nova amb les noves dimensions.
    return img2
