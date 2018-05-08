from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import vtk
import math
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import skimage
import skimage.io

# Function to nicely print segmentation results with
# colorbar showing class names
def discrete_matshow(data, labels_names=[], title=""):
    # get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin=np.min(data) - .5,
                      vmax=np.max(data) + .5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data), np.max(data) + 1))

    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
def imshow(image,num=1,titlename=None,colormap=plt.cm.gray):
  # display the 2D image
    fig=plt.figure(num)
  # Remove the first empty dimension
    image = np.squeeze(image)
    plt.imshow(image.astype(np.uint8),cmap=colormap)
    plt.title(titlename)
    plt.show()

def subplots(images,num,cols,figsize=(8,9), colormap=plt.cm.gray):
    #fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig=plt.figure(num=num,figsize=figsize)
    images=np.squeeze(images)
    ax=[]
    #plt.setp(axes.flat, xticks=[], yticks=[])
    Numimgs=images.shape[0]
    gs = gridspec.GridSpec(Numimgs // cols, cols)
    for i in range(Numimgs):
        row = (i // cols)
        col = i % cols
        ax.append(fig.add_subplot(gs[row, col]))
        img = images[i,:, :]
        #axlog[-1].set_title('markevery=%s' % str(case))
        #ax[-1].set_xscale('log')
        #ax[-1].set_yscale('log')
        ax[-1].imshow(img.astype(np.uint8), cmap=colormap)
        ax[-1].axis('off')
    plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
   # fig.subplots_adjust(bottom=0.05, right=0.95)

    plt.show()
def SuperPatchshow(testCord,PatchNorm,superpixel,slice_entro,labelslice,color='b',stride=32):
    from skimage.segmentation import mark_boundaries
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    #plt.gray()
    ax.imshow(mark_boundaries(PatchNorm, superpixel))
    figcolor=ax.imshow(slice_entro, cmap='rainbow',alpha=0.5)
    ax.imshow(labelslice, cmap='hot', alpha=0.5)
    fig.colorbar(figcolor, ax=ax)
   #ax.imshow(PatchNorm)
    ax.plot(np.transpose(np.array(testCord))[1],np.transpose(np.array(testCord))[0],'yo',lw=15)
    for c in range(len(testCord)):
         ax.add_patch(patches.Rectangle((testCord[c][1]-int(stride/2), testCord[c][0]-int(stride/2)),# (x,y)
                                                        stride,          # width
                                                        stride,          # heigh
                                                        fill=False,lw=5,edgecolor=color
                                                        ))

    ax.axis('off')
    plt.show()
def PixelPatchshow(testCord,PatchNorm,labelslice,color='b',stride=32):
    from skimage.segmentation import mark_boundaries
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    #plt.gray()
    ax.imshow(PatchNorm,cmap=plt.cm.gray)
    ax.imshow(labelslice, cmap='hot', alpha=0.5)

   #ax.imshow(PatchNorm)
    ax.plot(np.transpose(np.array(testCord))[1],np.transpose(np.array(testCord))[0],'yo',lw=15)
    # for c in range(len(testCord)):
    #      ax.add_patch(patches.Rectangle((testCord[c][1]-int(stride/2), testCord[c][0]-int(stride/2)),# (x,y)
    #                                                     stride,          # width
    #                                                     stride,          # heigh
    #                                                     fill=False,lw=5,edgecolor=color
    #                                                     ))

    ax.axis('off')
    plt.show()

class MiaViewer(object):
    """description of class"""
    def __init__(self):
     # self.input=input
     pass
    def SetInput(self,input):
        self._input=input
    def ColorMap(self):
        # Create a greyscale lookup table
        self._table = vtk.vtkLookupTable()
        self._table.SetRange(0, 5000) # image intensity range
        self._table.SetValueRange(0.0, 1.0) # from black to white
        self._table.SetSaturationRange(0.0, 0.0) # no color saturation
        self._table.SetRampToLinear()
        self._table.Build()
        # Map the image through the lookup table
        self._color = vtk.vtkImageMapToColors()
        self._color.SetLookupTable(self._table)
        self._color.SetInputConnection(self._input.GetOutputPort())
# Make a actor for rendering
    def MakeActors(self):
        actor = vtk.vtkImageActor()
        self.ColorMap()
        actor.GetMapper().SetInputConnection(self._color.GetOutputPort())
        return actor
    def Mia3DViewer(self):
        #*********rendering parameters setting using volume render***********
        volume=vtk.vtkVolume()
        mapper=vtk.vtkFixedPointVolumeRayCastMapper()
        #mapper=vtk.vtkVolumeRayCastMapper()
        mapper.SetInputConnection(self._input.GetOutputPort())
        #mapper.SetInputData(datareader.GetOutputData().GetOutput())
        vtkColor=vtk.vtkColorTransferFunction()
        opacityFun=vtk.vtkPiecewiseFunction()
        property =vtk.vtkVolumeProperty()
        property.SetColor(vtkColor)
        property.SetScalarOpacity(opacityFun)
        property.SetInterpolationTypeToLinear()
        volume.SetProperty(property)
        volume.SetMapper(mapper)
        vtkColor.AddRGBPoint( -3024, 0, 0, 0, 0.5, 0.0 )
        vtkColor.AddRGBPoint( -155, .55, .25, .15, 0.5, .92 )
        vtkColor.AddRGBPoint( 217, .88, .60, .29, 0.33, 0.45 )
        vtkColor.AddRGBPoint( 420, 1, .94, .95, 0.5, 0.0 )
        vtkColor.AddRGBPoint( 3071, .83, .66, 1, 0.5, 0.0 )
        opacityFun.AddPoint(-3024, 0, 0.5, 0.0 )
        opacityFun.AddPoint(-155, 0, 0.5, 0.92 )
        opacityFun.AddPoint(217, .68, 0.33, 0.45 )
        opacityFun.AddPoint(420,.83, 0.5, 0.0)
        opacityFun.AddPoint(3071, .80, 0.5, 0.0)
        mapper.SetBlendModeToComposite()
        property.ShadeOn()
        property.SetAmbient(0.1)
        property.SetDiffuse(0.9)
        property.SetSpecular(0.2)
        property.SetSpecularPower(10.0)
        property.SetScalarOpacityUnitDistance(0.8919)
        return volume
        # Create an actor
        #actor = vtk.vtkActor()
        #actor.SetMapper(mapper)
        #actor.GetProperty().SetColor(1.0,0.0,1.0)
    def Update(self):
        render=vtk.vtkRenderer()
        renWin=vtk.vtkRenderWindow()
        volume=self.Mia3DViewer()
        render.AddVolume(volume)
        renWin.AddRenderer(render)
        iren=vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        iren.Start()


