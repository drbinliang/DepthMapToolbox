'''
Created on 10 Nov 2015

@author: bliang03
'''
from depth_proc.depth_utils import loadDepthFile, showDepthData, getWorldCoordinates,\
    visualizePointCloud, getFrontDepthProjections, rotatePoints,\
    getDepthProjection
    
import math
from seq_representation.motion_history_rep import calDepthMHI,\
    rotateDepthSequence, calDepthMHT
import cv2

def run_depth_utils_example():
    """
        FUNC: examples of basic usage of depth utils
    """
#     # load binary depth file
#     depthFile = './depth_data/a01_s01_e01_sdepth.bin'
#     depthSequence = loadDepthFile(depthFile, fType = 'bin')

    # load png depth files in a given path
    depthFilePath = './depth_data/d_a01_s01_e01_sdepth/'
    depthSequence = loadDepthFile(depthFilePath, fType = 'png')
    
    for depthData in depthSequence:
        
        # show depth data
        showDepthData(depthData)
        
        # show 3d points
        points = getWorldCoordinates(depthData)
        visualizePointCloud(points)
        
        projData = getFrontDepthProjections(points, projSize = depthData.shape)
#         projImg = getDepthProjections(points)
        showDepthData(projData)
        
        r_alpha = math.pi / 2.     # angle around x-axis
        r_beta = 0   # angle around y-axis
        
        r_points = rotatePoints(points, r_alpha, r_beta)
        visualizePointCloud(r_points)
        
        projData = getDepthProjection(r_points)
        showDepthData(projData)


def run_seq_rep_example():
    """
        FUNC: examples of usage of sequence representation
    """
    # load binary depth file
    depthFile = './depth_data/a01_s01_e01_sdepth.bin'
    depthSequence = loadDepthFile(depthFile, fType = 'bin')
    
    dmht = calDepthMHT(depthSequence)
    
    # visualization
    cv2.imshow('front dmhi', dmht[0])
    cv2.imshow('top dmhi', dmht[1])
    cv2.imshow('side dmhi', dmht[2])
    cv2.waitKey()
        
    # any projections of frames in a sequence
    r_alpha = 0.
    r_beta = math.pi / 3.
    rProjSeq = rotateDepthSequence(depthSequence, r_alpha, r_beta)
    r_dmhi = calDepthMHI(rProjSeq)
    cv2.imshow('rotate projections', r_dmhi), cv2.waitKey()


# Example of usage   
if __name__ == '__main__':
#     run_type = 'depth_utils'
    run_type = 'seq_representation'
    
    if run_type == 'depth_utils':
        run_depth_utils_example()
    elif run_type == 'seq_representation':
        run_seq_rep_example()
