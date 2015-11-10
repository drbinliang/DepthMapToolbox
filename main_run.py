'''
Created on 10 Nov 2015

@author: bliang03
'''
from depth_utils import loadDepthFile, showDepthData, getWorldCoordinates,\
    visualizePointCloud, getFrontDepthProjections, rotatePoints,\
    getDepthProjections
    
import math

# Example of how to use   
if __name__ == '__main__':
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
        
        projImg = getFrontDepthProjections(points, projSize = depthData.shape)
#         projImg = getDepthProjections(points)
        showDepthData(projImg)
        
        r_alpha = math.pi / 2.     # angle around x-axis
        r_beta = 0   # angle around y-axis
        
        r_points = rotatePoints(points, r_alpha, r_beta)
        visualizePointCloud(r_points)
        
        projImg = getDepthProjections(r_points)
        showDepthData(projImg)