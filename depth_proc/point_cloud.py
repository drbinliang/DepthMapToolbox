'''
Created on 15 Sep 2014

@author: bliang03
'''

import vtk
from numpy import random

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        
    def visualize(self):
        
        # a renderer and render window
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        
        # an interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor() 
        renderWindowInteractor.SetRenderWindow(renderWindow)
        
        #add the actors to the scene
        renderer.AddActor(self.vtkActor)
        renderer.SetBackground(255, 255, 255) # Background white
#         renderer.SetBackground(0, 0, 0)
          
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, 0)
         
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        #  The axes are positioned with a user transform
        axes.SetUserTransform(transform)
        
#         renderer.AddActor(axes)
 
        renderer.ResetCamera()
        
        # Begin Interaction
        renderWindow.SetSize(400, 400)
        renderWindow.Render()
        renderWindowInteractor.Start()


# example
if __name__ == '__main__':
    pointCloud = VtkPointCloud()
    for k in xrange(1000):
        point = 20*(random.rand(3)-0.5)
        pointCloud.addPoint(point)
    pointCloud.addPoint([0,0,0])
    pointCloud.addPoint([0,0,0])
    pointCloud.addPoint([0,0,0])
    pointCloud.addPoint([0,0,0])
    pointCloud.visualize()
    
    
#     # Renderer
#     renderer = vtk.vtkRenderer()
#     renderer.AddActor(pointCloud.vtkActor)
#     renderer.SetBackground(.2, .3, .4)
#     renderer.ResetCamera()
#     
#     # Render Window
#     renderWindow = vtk.vtkRenderWindow()
#     renderWindow.AddRenderer(renderer)
#     
#     # Interactor
#     renderWindowInteractor = vtk.vtkRenderWindowInteractor()
#     renderWindowInteractor.SetRenderWindow(renderWindow)
#     
#     # Begin Interaction
#     renderWindow.Render()
#     renderWindowInteractor.Start()