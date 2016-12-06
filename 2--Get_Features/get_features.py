# -*- coding: cp1252 -*-
from get_params import get_params
import sys
#from rootsift import RootSIFT
import os, time
import numpy as np
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def get_features(params,pca=None,scaler=None):

    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    # Initialize keypoint detector and feature extractor
    #detector, extractor = init_detect_extract(params)

    # Initialize feature dictionary
    features = {}

    # Get trained codebook
    km = pickle.load(open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'rb'))

    for image_name in image_list:
        image_name=image_name.replace('\n','')
        # Read image
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',image_name))

        # Resize image
        im = resize_image(params,im)

        # Extract local features
        feats = image_local_features(im)

        if feats is not None:

            if params['normalize_feats']:
                feats = normalize(feats)

            # If we scaled training features
            if scaler is not None:
                scaler.transform(feats)

            # Whiten if needed
            if pca is not None:

                pca.transform(feats)

            # Compute assignemnts
            assignments = get_assignments(km,feats)

            # Generate bow vector
            feats = bow(assignments,km)
        else:
            # Empty features
            feats = np.zeros(params['descriptor_size'])

        # Add entry to dictionary
        features[image_name] = feats


    # Save dictionary to disk with unique name
    save_file = os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             params['split'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p')

    pickle.dump(features,open(save_file,'wb'))


def resize_image(params,im):

    # Get image dimensions
    height, width = im.shape[:2]

    # If the image width is smaller than the proposed small dimension, keep the original size !
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # Resize and return new image
    return cv2.resize(im,dim)

'''
class RootSIFT:
    def __init__(self):
    	# initialize the SIFT feature extractor
    	self.extractor = cv2.DescriptorExtractor_create("SIFT")

    def compute(self, image, kps, eps=1e-7):
    	# compute SIFT descriptors
    	(kps, descs) = self.extractor.compute(image, kps)

    	# if there are no keypoints or descriptors, return an empty tuple
    	if len(kps) == 0:
    	   return ([], None)

    	# apply the Hellinger kernel by first L1-normalizing and taking the
    	# square-root
    	descs /= (descs.sum(axis=1, keepdims=True) + eps)
    	descs = np.sqrt(descs)

        return kps, descs

def image_local_features(im,detector,extractor):


    #Extract local features for given image


    positions = detector.detect(im,None)
    positions, descriptors = extractor.compute(im,positions)

    return descriptors
'''

def image_local_features(image):
    #llegim la imatge:
    #img = cv2.imread(image)
    #Cambiem la mida de la imatge:
    if not image is None:

        #linea que soluciona un bug de opencv a python3
        #cv2.ocl.setUseOpenCL(False)

        # Creem l'objecte ORB que tindrà 200k keypoints. (Perametre que podem modificar per no saturar el programa)
        orb = cv2.ORB(200000)

        # Detectem els keypoints:
        kp = orb.detect(image,None)

        # Calculem els descriptors amb els keypoints trobats.
        kp, des = orb.compute(image, kp)

        # la sortida de la funció serà els descriptors
        return des

"""def init_detect_extract(params):

    '''
    Initialize detector and extractor from parameters
    '''
    if params['descriptor_type'] == 'RootSIFT':

        extractor = RootSIFT()



    else:

        extractor = cv2.DescriptorExtractor_create(params['descriptor_type'])

    detector = cv2.FeatureDetector_create(params['keypoint_type'])

    return detector, extractor
"""

def stack_features(params):

    '''
    Get local features for all training images together
    '''

    # Init detector and extractor
    #detector, extractor = init_detect_extract(params)

    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    X = []
    for image_name in image_list:
        image_name=image_name.replace('\n','') #Afegim el replace per ignorar el caràcter /n

        # Read image
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',image_name))

        # Resize image
        im = resize_image(params,im)

        #feats = image_local_features(im,detector,extractor)
        feats=image_local_features(im)
        # Stack all local descriptors together

        if feats is not None:
            if len(X) == 0:

                X = feats
            else:
                X = np.vstack((X,feats))

    if params['normalize_feats']:
        X = normalize(X)

    if params['whiten']:

        pca = PCA(whiten=True)
        pca.fit_transform(X)

    else:
        pca = None

    # Scale data to 0 mean and unit variance
    if params['scale']:

        scaler = StandardScaler()

        scaler.fit_transform(X)
    else:
        scaler = None

    return X, pca, scaler

def train_codebook(params,X):

    # Init kmeans instance
    km = MiniBatchKMeans(params['descriptor_size'])

    # Training the model with our descriptors
    km.fit(X)

    # Save to disk
    pickle.dump(km,open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'wb'))

    return km

def get_assignments(km,descriptors):

    assignments = km.predict(descriptors)

    return assignments


def bow(assignments,km):

    # Initialize empty descriptor of the same length as the number of clusters
    descriptor = np.zeros(np.shape(km.cluster_centers_)[0])

    # Build vector of repetitions
    for a in assignments:

        descriptor[a] += 1

    # L2 normalize
    descriptor = normalize(descriptor)

    return descriptor



if __name__ == "__main__":

    params = get_params()

    # Change to training set
    params['split'] = 'train'

    print "Stacking features together..."
    # Save features for training set
    t = time.time()
    X, pca, scaler = stack_features(params)
    print "Done. Time elapsed:", time.time() - t
    print "Number of training features", np.shape(X)

    print "Training codebook..."
    t = time.time()
    train_codebook(params,X)
    print "Done. Time elapsed:", time.time() - t

    print "Storing bow features for train set..."
    t = time.time()
    get_features(params, pca,scaler)
    print "Done. Time elapsed:", time.time() - t

    params['split'] = 'val'

    print "Storing bow features for validation set..."
    t = time.time()
    get_features(params)
    print "Done. Time elapsed:", time.time() - t
