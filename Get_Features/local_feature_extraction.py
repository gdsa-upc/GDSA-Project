
import numpy
import cv2
from matplotlib import pyplot as plt

#Definim la funció resize_image per cambiar la mida de les imatges massa grans mantenint
#les proporcions (aspect ratio) height/width.
def resize_image(params,img):
    #Obtenim les dimensions de l'imatge amb la funció de numpy, shape.
    height, width = img.shape[:2]

    # En cas que la mida de la imatge sigui més petita que el parametre max_size
    #(mida maxima que especifiquem al invocar la funció), manté la mida original.
    #Si la mida de la imatge és més gran, passa a ser de la mida max_size.
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # La funció retorna una imatge nova amb les noves dimensions.
    return cv2.resize(img,dim)


def local_feature_extraction(params,image):
    #llegim la imatge:
    img = cv2.imread(image,0)
    #Cambiem la mida de la imatge:
    img = resize_image(params,img)
    #linea que soluciona un bug de opencv
    cv2.ocl.setUseOpenCL(False)

    # Creem l'objecte ORB que tindrà 200k keypoints. (Perametre que podem modificar per no saturar el programa)
    orb = cv2.ORB_create(200000)

    # Detectem els keypoints:
    kp = orb.detect(img,None)

    # Calculem els descriptors amb els keypoints trobats.
    kp, des = orb.compute(img, kp)

    # la sortida de la funció serà els descriptors
    return des
