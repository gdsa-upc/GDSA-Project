# -*- coding: utf-8 -*-
import numpy as np
import cv2
from RootSIFT import RootSIFT

def get_local_features(params,image):
    img = cv2.imread(image)
    #Fem un resize de la imatge
    img = resize_image(params,img) #Si volem un tamany específic ho fem aixi: img = cv2.resize(img, (250, 250)) 

    # Initiate STAR detector
    orb = cv2.ORB(200000)
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    # extract and compute RootSIFT descriptors
    extractor = RootSIFT()
    kp,des= extractor.compute(gray,kp,params['descriptor_size'])

    return des


def resize_image(params,im):
    # Get image dimensions
    height, width = im.shape[:2]

    # If the image width is smaller than the proposed small dimension, keep the original size !
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # Resize and return new image
    return cv2.resize(im,dim)