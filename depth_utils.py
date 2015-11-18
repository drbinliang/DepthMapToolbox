'''
Created on 29 Oct 2015

@author: bliang03
'''
import os
import struct
import numpy as np
import cv2
from utils import mat2gray, findBoxRegion
from point_cloud import VtkPointCloud
import math

def loadDepthFile(depthFile, fType = 'bin'):
    """ 
        FUNC: load depth file
        
        PARAM:
            depthFile: depth file path
            fType: depth file type, 'bin' (default) or 'png'
            
        RETURN:
            depthSequence: a list of depth frame
    """
    
    def loadBinDepthFile(depthFile):
        """
            FUNC: load binary depth file with extension of '.bin'
            
            PARAM:
                depthFile: depth file path
                
            RETURN: 
                depthSequence: a list of depth frame 
        """
        with open(depthFile, 'rb') as f:
            # bin file with 'ubit32' (4 bytes)
            n_frames = struct.unpack("I", f.read(4))[0]  # 'I' means unsigned int
            n_cols = struct.unpack("I", f.read(4))[0]
            n_rows = struct.unpack("I", f.read(4))[0]
            
            depthSequence = []
            
            for n in xrange(n_frames):
                depthData = np.zeros((n_rows, n_cols), dtype=np.double)
                for i in xrange(n_rows):
                    for j in xrange(n_cols):
                        depthData[i, j] = struct.unpack("I", f.read(4))[0]
                        
                depthSequence.append(depthData)
            
        f.close()   
        
        return depthSequence 
    
    def loadPngDepthFile(depthFilePath):
        """
            FUNC: load depth file with extension of '.png'
            
            PARAM: 
                depthFilePath: file path where png depth files are
                
            RETURN: 
                depthSequence: a list of depth frame 
        """
        pngFiles = [f for f in os.listdir(depthFilePath) if f.endswith('.png')] 
        depthSequence = []
        
        for pngFile in pngFiles:
            depthData = cv2.imread(os.path.join(depthFilePath, pngFile), cv2.IMREAD_UNCHANGED)
            depthSequence.append(depthData)
            
        return depthSequence
    
    
    if fType.lower() == 'bin':
        # load binary depth file
        depthSequence = loadBinDepthFile(depthFile)
    elif fType.lower() == 'png':
        # load png depth file in a given path
        depthFilePath = depthFile
        depthSequence = loadPngDepthFile(depthFilePath)
    else:
        pass
    
    return depthSequence


def getWorldCoordinates(depthData):
    """ 
        FUNC: get world coordinates from the given depth data
        
        PARAM:
            depthData: depth data
            
        RETURN: 
            points [n_points * 3]
    """
    fx = 580.0  # focal length x
    fy = 580.0  # focal length y
    
    height, width = depthData.shape
    
    cx = width / 2.
    cy = height / 2.
    
#     xwList = []
#     ywList = []
#     zwList = []
    points = np.zeros((height * width, 3))
    i = 0
    for rowIdx in xrange(height):
        for colIdx in xrange(width):
            zw = depthData[rowIdx, colIdx]
            
            xw = (colIdx - cx) * zw / fx
            yw = (rowIdx - cy) * zw / fy
            
#             xwList.append(xw)
#             ywList.append(yw)
#             zwList.append(zw)
            
            points[i,:] = np.array([xw, yw, zw])
            i += 1
     
#     points[:,1] = -1. * points[:,1] # upside down along y-direction
# #     points[:,0] = -1. * points[:,0]
#     points[:,2] = -1. * points[:,2]
    
    return points


def getFrontDepthProjections(points, projSize):
    """
        FUNC:
            project point clouds onto image plane
        PARAM:
            points: point clouds, [n_points, 3]
            projSize: the size of the image plane to be projected
                      [width, height]
        RETURN:
            projImg: projected image
    """
    fx = 580.0  # focal length x
    fy = 580.0  # focal length y
    
    height, width = projSize
    cx = width / 2.
    cy = height / 2.
    
    n_points, _ = points.shape
    
    # projImg initialization
    projData = np.zeros((height, width))    
    for n in xrange(n_points):
        xw, yw, zw = points[n, :]
        if zw != 0:
            colIdx = math.floor((xw * fx) / zw + cx)
            rowIdx = math.floor((yw * fy) / zw + cy)
            projData[rowIdx, colIdx] = zw
            if colIdx < 0 or rowIdx < 0:
                print 'WARNING: index is out of range when doing projection '
    
    return projData


def getDepthProjection(points, isCrop = True):
    """
        FUNC: get projections by counting the number of 3D points
        
        PARAM:
            points: point clouds
            crop: crop the ROI or not
            
        RETURN:
            resultProjImg: the projection image (range: 0-255)
    """
    tmp_points = points.copy()
    min_xw = np.min(tmp_points[:,0])
    min_yw = np.min(tmp_points[:,1])
     
    tmp_points[:,0] -= min_xw
    tmp_points[:,1] -= min_yw
     
#     width = math.ceil(np.max(tmp_points[:,0])) + 1
#     height = math.ceil(np.max(tmp_points[:,1])) + 1

    width = 800
    height = 800
     
    n_points, _ = tmp_points.shape
     
    projData = np.zeros((height, width))
    for n in xrange(n_points):
        [xw, yw, zw] = tmp_points[n, :]
        if zw != 0:
            rowIdx = math.floor(yw)
            colIdx = math.floor(xw)
            projData[rowIdx, colIdx] += 1
     
    projImg = mat2gray(projData)  # scale to 0-255
    
    # post processing
    # closing (Dilation followed by Erosion)
    kernel = np.ones((3, 3), np.uint8)
    projImg1 = cv2.morphologyEx(projImg, cv2.MORPH_CLOSE, kernel)
    projImg2 = cv2.equalizeHist(projImg1)
    
    if isCrop:
        boxRegion = findBoxRegion(projImg2)
        top, bottom, left, right = boxRegion
        resultProjImg = projImg2[top:bottom, left:right]
    else:
        resultProjImg = projImg2.copy()
        
    return resultProjImg
    


def visualizePointCloud(points):
    """ 
        FUNC: visualization of point clouds
        
        PARAM:
            points: point clouds
            
        RETURN: 
            
    """
    # 3d point cloud visualization
    numPoints, _ = points.shape
    zMin = points[:,2].min()
    zMax = points[:,2].max()
    pointCloud = VtkPointCloud(zMin = zMin, zMax = zMax)
    for i in xrange(numPoints):
        point = points[i, :]
        if (point != 0).any():
            # ignore the origin point
            pointCloud.addPoint(point)
     
    pointCloud.visualize()


def showDepthData(depthData, isColorMap = True):
    """
        FUNC: show depth data
        
        PARAM:
            depthData: raw depth data
        
        RETURN:
            
    """
    depthImage = mat2gray(depthData)
    
    if isColorMap:
        depthImage = cv2.applyColorMap(depthImage, cv2.COLORMAP_JET)
    
    cv2.imshow('', depthImage)
    cv2.waitKey()
    

def rotatePoints(points, r_alpha, r_beta):
    """
        FUNC: rotate 3D point clouds
        
        PARAM:
            points: point clouds
            r_alpha: angle around y-axis
            r_beta: angle around x-axis
            
        RETURN:
            r_points: rotated points
        
    """

# #     Method in paper "ConvNets-Based Action Recognition from Depth Maps
# #     through Virtual Cameras and Pseudocoloring"    
#     n_points, _ = points.shape
#     r_points = np.zeros((n_points, 3))
#     for n in xrange(n_points):
#         [xw, yw, zw] = points[n,:]
#          
#         ry = np.array((
#                 (1,               0,               0,  0),
#                 (0, math.cos(r_alpha), -math.sin(r_alpha), zw * math.sin(r_alpha)),
#                 (0, math.sin(r_alpha),  math.cos(r_alpha), zw * (1 - math.cos(r_alpha))),
#                 (0,               0,                0, 1) 
#             ))
#         rx = np.array((
#                 (math.cos(r_beta),  0,  math.sin(r_beta), -zw * math.sin(r_beta)),
#                 (0,               1,               0, 0),
#                 (-math.sin(r_beta), 0,  math.cos(r_beta), zw * (1 - math.cos(r_beta))),
#                 (0,               0,               0, 1) 
#             ))
#   
#         p = np.array((xw, yw, zw, 1)).reshape((4,1))
#           
#         r1 = np.dot(ry, rx)
#         r2 = np.dot(r1, p)
#           
#         r_point = r2.reshape((1, 4))[0, :3]
#           
#         r_points[n, :] = r_point
        
    # Our method
    ry = np.identity(4)
    ry[1,1] = math.cos(r_alpha)
    ry[1,2] = -math.sin(r_alpha)
    ry[2,1] = math.sin(r_alpha)
    ry[2,2] = math.cos(r_alpha)
     
    rx = np.identity(4)
    rx[0, 0] = math.cos(r_beta)
    rx[0, 2] = math.sin(r_beta)
    rx[2, 0] = -math.sin(r_beta)
    rx[2, 2] = math.cos(r_beta)
    
    n_points, _ = points.shape 
    
    # uncomment to move to origin for rotation
#     t_points = np.hstack((-1 * points, np.ones((n_points, 1))))
    t_points = np.hstack((points, np.ones((n_points, 1))))
    r1 = np.dot(ry, t_points.transpose())
    r2 = np.dot(rx, r1)
    t_r_points = r2.transpose()
    
    r_points = t_r_points[:, :3]
    return r_points    
        

    
