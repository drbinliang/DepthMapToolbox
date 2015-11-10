'''
Created on 30 Oct 2015

@author: bliang03
'''
import numpy as np


def mat2gray(mat):
    """ 
        FUNC: convert matrix to gray image (0-255)
        
        PARAM: 
            mat: matrix to be converted
            
        RETURN: 
            grayImg: converted gray image from the given mat
            
    """
    minItem = mat.min()
    maxItem = mat.max()
    
    grayImg = mat.copy()
    if (maxItem - minItem) != 0:
        tmpMat = (mat - minItem) / float(maxItem - minItem) # scale to [0-1]
        grayImg = tmpMat * 255
        grayImg = grayImg.astype(np.uint8)
    
    return grayImg


def findBoxRegion(image):
    """ 
        FUNC: find the box region of given image 
        
        PARAM:
            image: the given image
            
        RETURN:
            boxRegion: in the form of [top, bottom, left, right]
    """
    # top
    rows, cols = image.shape
    top = 0
    for r in xrange(rows):
        row = image[r, :]
        if sum(row) > 0:
            top = r
            break
        
    # bottom
    bottom = rows - 1
    for r in xrange(rows-1, -1, -1):
        row = image[r, :]
        if sum(row) > 0:
            bottom = r
            break;
        
    # left
    left = 0
    for c in xrange(cols):
        col = image[:, c]
        if sum(col) > 0:
            left = c
            break
    
    # right
    right = cols - 1
    for c in xrange(cols-1, -1, -1):
        col = image[:, c]
        if sum(col) > 0:
            right = c
            break
        
    boxRegion = [top, bottom, left, right]
    return boxRegion