'''
Created on 18 Nov 2015

@author: bliang03
'''
import cv2
import numpy as np
from skimage.feature._hog import hog
import csv

def extractPHOGFeature(image, levels = (0, 3)):
    ''' 
        FUNC: pyramid hog calculation
        PARAM: 
            image: input image where features are extracted
            levels: tuple, represents start level and end level
        RETURN:
            pHOGVec: phog feature vector
    '''
    height, width = image.shape
    orientationNum = 9
    pHOGVec = np.array([])
    base = 2
    
    startLevel, endLevel = levels
    
    for l in xrange(startLevel, endLevel + 1):
        gridNum = base ** l
        
        # dimentionality of hog feature vector is 
        # orientations * gridNum * gridNum
        featureVector, hogImage = hog(image, orientations = orientationNum, 
                                  pixels_per_cell = (width/gridNum, height/gridNum),
                                  cells_per_block=(1, 1), visualise = True)
        
#         # visualization of hog image
#         cv2.imshow('', hogImage), cv2.waitKey()
        
        if pHOGVec.size == 0:
            pHOGVec = featureVector.copy()
        else:
            pHOGVec = np.concatenate((pHOGVec, featureVector))
    
#     # write feature vector into a csv file
#     tmpPHOGVec = pHOGVec.copy()
#     n_dim = tmpPHOGVec.size
#     tmpPHOGVec = tmpPHOGVec.reshape(n_dim, 1)
#     with open('./phog_vec.csv', 'wb') as f:
#         writer = csv.writer(f)
#         writer.writerows(tmpPHOGVec.tolist())
#     f.close()
    
    return pHOGVec