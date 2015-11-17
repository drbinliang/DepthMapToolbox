'''
Created on 23 Sep 2014

@author: bliang03
'''
import numpy as np
import cv2
from utils import mat2gray, split_list

def depthFrameDiff(currImg, preImg, motion_thresh):
    ''' frame difference '''
    diff = cv2.absdiff(currImg, preImg).astype(np.int32)
    motionImg = diff.copy()
    motionImg[np.where(diff >= motion_thresh)] = 1
    motionImg[np.where(diff < motion_thresh)] = 0
    
    return motionImg


def calDepthMHI(frames, motion_thresh = 10, stride = 1):
    """ calculate DMHI from given frames """
    
    duration = len(frames)
    firstFrm = frames[0]
    firstFrm = cv2.GaussianBlur(firstFrm, (3, 3), 0)    # Gaussian blur
            
    height, width = firstFrm.shape
    
    # a set of DMHIs
    D_MHIs = []
    D_MHIs.append(np.zeros((height, width)).astype(np.int32))
    
    for i in xrange(1, duration, stride):
        prevFrmIndex = i - stride
        
        if (prevFrmIndex >= 0):
            # current image
            currFrm = frames[i]
            currFrm = cv2.GaussianBlur(currFrm, (3, 3), 0)    # Gaussian blur
            
            # previous image
            prevFrm = frames[prevFrmIndex]
            prevFrm = cv2.GaussianBlur(prevFrm, (3, 3), 0)    # Gaussian blur
            
            # frame difference
            motionImg = depthFrameDiff(currFrm, prevFrm, motion_thresh)
            
            # DMHI
            # if D == 1
            DMHI = motionImg.copy()
            DMHI[np.where(motionImg == 1)] = duration
            
            # otherwise
            tmp = np.maximum(0, D_MHIs[-1] - 1)
            idx = np.where(motionImg != 1)
            DMHI[idx] = tmp[idx]
            
            D_MHIs.append(DMHI)
            
            # save mhi image to file
    #         dmhiImg = mat2gray(DMHI)
    #         cv2.imwrite('mhi_%i.png' %i, dmhiImg)
        
        
    # get the result    
    finalDMHI = D_MHIs[-1]
    
    # convert to image
    dmhiImg = mat2gray(finalDMHI)
    dmhiImg = cv2.GaussianBlur(dmhiImg, (3, 3), 0)    # Gaussian blur
    
    return dmhiImg


# def calWinDepthMHIList(frames, winSize, winStep):
#     """ calculate dmhi list using sliding window """
#     dmhiList = []
#     for i in xrange(0, len(frames), winStep):
#         startIdx = i
#         endIdx = i + winSize - 1
#         if endIdx <= len(frames) - 1:
# #             print "startIdx: %d, endIdx: %d" % (startIdx, endIdx)
#             winFrames = frames[startIdx:endIdx]
#             dmhiImg = calDepthMHI(winFrames)
#             dmhiList.append(dmhiImg)
#             
#     return dmhiList
# 
# 
# def calPyramidDepthMHIList(frames, dHOGTemporalPyramids):
#     """ calculate temporal Pyramid Depth MHI (PDMHI)  list"""
#     pdmhiList = []
#     for l in dHOGTemporalPyramids:
#         wantedParts = l
#         segmentedList = split_list(frames, wantedParts)
#         
#         for segment in segmentedList:
#             tmpFrames = segment
#             dmhiImg = calDepthMHI(tmpFrames)
#             
#             pdmhiList.append(dmhiImg)
#         
#     return  pdmhiList   
#     
# 
# def cropAndResizeImageList1(imageList, newSize = (128, 128)):
#     """ crop the image list and resize them (overall)"""
#     firstImg = imageList[0]
#     finalTop, finalBottom, finalLeft, finalRight = findBoxRegion(firstImg)
#     
#     for i in xrange(1, len(imageList)):
#         currImg = imageList[i]
#         currTop, currBottom, currLeft, currRight = findBoxRegion(currImg)
#         
#         if currTop < finalTop:
#             finalTop = currTop
#         
#         if currBottom > finalBottom:
#             finalBottom = currBottom
#         
#         if currLeft < finalLeft:
#             finalLeft = currLeft
#             
#         if currRight > finalRight:
#             finalRight = currRight
#             
#     postprocessImageList = []
#     for image in imageList:
#         cropedImage = image[finalTop:finalBottom, finalLeft:finalRight]
#         resizedImage = cv2.resize(cropedImage, newSize)
#         postprocessImageList.append(resizedImage)
#         
# #         # show image
# #         cv2.imshow('', resizedImage)
# #         cv2.waitKey()
#     
#     return postprocessImageList
#         
#     
# def findBoxRegion(image):
#     """ find the box region of given image """
#     # top
#     rows, cols = image.shape
#     top = 0
#     for r in xrange(rows):
#         row = image[r, :]
#         if sum(row) > 0:
#             top = r
#             break
#         
#     # bottom
#     bottom = rows - 1
#     for r in xrange(rows-1, -1, -1):
#         row = image[r, :]
#         if sum(row) > 0:
#             bottom = r
#             break;
#         
#     # left
#     left = 0
#     for c in xrange(cols):
#         col = image[:, c]
#         if sum(col) > 0:
#             left = c
#             break
#     
#     # right
#     right = cols - 1
#     for c in xrange(cols-1, -1, -1):
#         col = image[:, c]
#         if sum(col) > 0:
#             right = c
#             break
#         
#     return top, bottom, left, right
# 
# 
# def cropAndResizeImageList2(imageList, newSize = (128, 128)):
#     """ crop ROI from an image (one by one) """
#     postprocessImageList = []
#     for image in imageList:
#         top, bottom, left, right = findBoxRegion(image)
#         cropedImage = image[top:bottom, left:right]
#         resizedImage = cv2.resize(cropedImage, newSize)
#         postprocessImageList.append(resizedImage)
#         
#     return postprocessImageList
# 
# 
# def cropAndResizeImageList3(imageList, maxImSize = 300):
#     """ crop ROI from an image (one by one) """
#     postprocessImageList = []
#     for image in imageList:
#         top, bottom, left, right = findBoxRegion(image)
#         cropedImage = image[top:bottom, left:right]
#         resizedImage = resizeImage(cropedImage, maxImSize)
#         postprocessImageList.append(resizedImage)
#         
#     return postprocessImageList
# 
# 
# def resizeImage(originalImg, maxImSize):
#     """ 
#         maxImSize    -maximum size of the input image
#     """
#     nCols, nRows = originalImg.shape
#     
#     if max(nCols, nRows) > maxImSize:
#         fx = float(maxImSize)/max(nCols, nRows)
#         resizedImg = cv2.resize(originalImg, (0,0), fx=fx, fy=fx, interpolation = cv2.INTER_CUBIC) 
#     else:
#         resizedImg = originalImg.copy()
#         
#     return resizedImg
# 
# 
# def computeHMHT(imageList):
#     """ compute hierarchical motion history templates """
#     strides = config.strides
#     hmhtList = []
#     for stride in strides:
#         dmhiImg = calDepthMHI(imageList, motion_thresh = 10, stride = stride)
#         
#         # find ROI
#         top, bottom, left, right = findBoxRegion(dmhiImg)
#         cropedImage = dmhiImg[top:bottom, left:right]
#         resizedImage = cv2.resize(cropedImage, (128, 128))
#         hmhtList.append(resizedImage)
#         
#     return hmhtList
        
