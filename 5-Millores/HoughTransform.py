import cv2
import numpy as np
from get_params import get_params
import matplotlib.pyplot as plt
import os
from get_features import resize_image

def HoughTransform (params):

    img=cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images','168-2743-15592.jpg'))
    img=resize_image(params,img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray,50,150,apertureSize = 3) # Troba els extrems de la imatge
    lines=cv2.HoughLinesP(dst, 1, np.pi/180, 50, 1000, 1 ) #Threshold, minilinelenght, maxline gap
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('Result',img)
    cv2.waitKey()
params=get_params()
HoughTransform(params)
