from get_local_features_ORB import get_local_features_orb
from get_params import get_params
from drawmatches import drawMatches
import cv2
import os
import matplotlib.pyplot as plt
params = get_params()
i1 = cv2.imread('/'.join([params['arrel_entrada'],params['bd_imatges'],'train','images','3607-12343-7554.jpg']),0)
i2 = cv2.imread('/'.join([params['arrel_entrada'],params['bd_imatges'],'train','images','14496-8883-17763.jpg']),0)
k1,des1=get_local_features_orb('3607-12343-7554.jpg')
k2,des2=get_local_features_orb('14496-8883-17763.jpg')
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches,key=lambda val: val.distance)
print len(matches)
out = drawMatches(i1, k1, i2, k2, matches[:20])
plt.imshow(out)
plt.show()