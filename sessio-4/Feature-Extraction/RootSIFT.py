import numpy as np
import cv2
 
class RootSIFT:
	def __init__(self):
		#Inicialitzacio del self.extractor
		self.extractor = cv2.DescriptorExtractor_create("SIFT")
 
	def compute(self, image, kps, eps=1e-7):
		#Calcul dels descriptors SIFT
		(kps, descs) = self.extractor.compute(image, kps)
 
		#Sino hi ha kp, retorna un vector buit
		if len(kps) == 0:
			return ([], None)
 
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
 
		#Retorn dels keypoints i dels descriptors
		return (kps, descs)