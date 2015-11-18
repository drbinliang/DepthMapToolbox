'''
Created on 23 Sep 2014

@author: bliang03
'''
import numpy as np
import cv2
from utils import mat2gray, split_list, findBoxRegion
from depth_utils import getWorldCoordinates, rotatePoints, getDepthProjection
import math

def depthFrameDiff(currImg, preImg, motion_thresh):
    ''' 
        FUNC: calculate motion regions given two frames
        PARAM:
            currImg: current image
            preImg: previous image
            motion_thresh: threshold for detection of motion region
        RETURN:
            motionImg: motion image which encodes the motion regions 
    '''
    diff = cv2.absdiff(currImg, preImg).astype(np.int32)
    motionImg = diff.copy()
    motionImg[np.where(diff >= motion_thresh)] = 1
    motionImg[np.where(diff < motion_thresh)] = 0
    
    return motionImg


def calDepthMHI(frames, motion_thresh = 10, stride = 1, isCrop = True):
    """ 
        FUNC: Calculate DMHI from given sequence (a list of frames)
        PARAM:
            frames: a list of frames
            motion_thresh: threshold for detection of motion region
            stride: stride of calculation of difference between frames
            isCrop: crop ROI or not
        RETURN:
            dmhi: depth motion history image
    """
    
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
    dmhi = mat2gray(finalDMHI)
    dmhi = cv2.GaussianBlur(dmhi, (3, 3), 0)    # Gaussian blur
    
    # crop or not
    if isCrop:
        boxRegion = findBoxRegion(dmhi)
        top, bottom, left, right = boxRegion
        resultDmhi = dmhi[top:bottom, left:right]
    else:
        resultDmhi = dmhi.copy()
    
    return resultDmhi



def calDepthMHT(frames, motion_threshs = [10, 10, 10]):
    """
        FUNC: Calculate DMHT from given sequence (a list of depth frames)
        PARAM:
            frames: a list of depth frames
            motion_thresh: a list of thresholds for detection of motion region
        RETURN:
            dmht: a list of dmhi (front, top and side) 
    """
    dmht = []
    
    # front view
    dmhi = calDepthMHI(frames, motion_thresh = motion_threshs[0])
    dmht.append(dmhi)
    
    # top view
    r_alpha = math.pi / 2.  # angle around x
    r_beta = 0. # angle around y
    # top projections of frames in a sequence
    topProjSeq = rotateDepthSequence(frames, r_alpha, r_beta)
    top_dmhi = calDepthMHI(topProjSeq, motion_thresh = motion_threshs[2])
    dmht.append(top_dmhi)
    
    # side view
    r_alpha = 0.    # angle around x
    r_beta = math.pi / 2.   # angle around y
    # side projections of frames in a sequence
    sideProjSeq = rotateDepthSequence(frames, r_alpha, r_beta)
    side_dmhi = calDepthMHI(sideProjSeq, motion_thresh = motion_threshs[1])
    dmht.append(side_dmhi)
    
    return dmht


def rotateDepthSequence(frames, r_alpha, r_beta):
    """
        FUNC: rotate depth frames in a sequence
        PARAM:
            frames: a list of frames
            r_alpha: angle around y-axis
            r_beta: angle around x-axis
        RETURN:
            r_frames: a list of rotated frames
    """
    r_frames = []
    for depthData in frames:
        points = getWorldCoordinates(depthData)
        r_points = rotatePoints(points, r_alpha, r_beta)
        projData = getDepthProjection(r_points, isCrop=False)
        r_frames.append(projData)
    
    return r_frames


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
        
