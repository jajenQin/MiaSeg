import vtk
import os
import SimpleITK as itk
from numpy import *
import scipy as scp
import scipy.misc
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class MiaDataPreprocess(object):
    """description of class"""
    def __init__(self):
     # self.input=input
       # pass
     self._Batch2D=[]
     self._Batch3D=[]
####Set array input
    def SetArray(self,array):
      self._array=array
####Get array data
    def GetArray(self):
      return self._array
####Get batch array data
    def GetBatch2DArray(self):
      return self._Batch2D
####Get batch 3D array data
    def GetBatch3DArray(self):
      return self._Batch3D

 # *********Batch 2D image data from a series of volume which are extrated from volume data
    def Batch2DVolume(self, vol, direction='Axial'):
          ###Different slice direction data can be got and batch process
          Slicelist = []
          labellist=[]
          if direction == 'Axial':
              for slice in arange(vol['volume'].shape[0]):

                  Slicelist.append(vol['volume'][slice, :, :])
                  labellist.append(vol['label'][slice, :, :])


          if direction == 'Sagittal':

              for slice in arange(vol['volume'].shape[2]):
                  Slicelist.append(vol['volume'][:,:,slice])
                  labellist.append(vol['label'][:, :,slice])

          if direction == 'Coronal':
              # self._Batch2D=self._array[:,0,:].ravel()
              for slice in arange(vol['volume'].shape[1]):
                  Slicelist.append(vol['volume'][:,slice,:])
                  labellist.append(vol['label'](shape)[:,slice,:])
          print(Slicelist)
          return {'slice':array(Slicelist),'label':array(labellist)}

  # *********Slice 2D image data from a 3D volume data
    def Slice2DArray_salient(self, vol,label,axis='Axial'):
        """
        Different slice direction data can be got and slice processing
        Parameters
          Inputs:
            axis:The slice Axis
        Return:
            a list of slice at axis
        """
        Slicelist=[]
        labellist=[]
        if axis == 'Axial':
             for slice in arange(vol.shape[0]):
                Batch =vol[slice, :, :]
                labelslice=label[slice,:,:]

                if slice==0:
                   Slicelist.append(Batch)
                   labellist.append(labelslice)
                else:
                  corelation=pearsonr(array(Batch).ravel(),array(Slicelist)[len(Slicelist)-1].ravel())[0]
                  print('The patch correlation cofficient:%f'%corelation)
                  if corelation<0.9:
                     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
                     #
                     # axes[0].imshow(array(Slicelist)[len(Slicelist)-1],cmap=plt.cm.gray)
                     # axes[0].imshow(array(labellist)[len(labellist)-1],alpha=0.5,cmap=plt.cm.gray)
                     #
                     # axes[1].imshow(Batch,cmap=plt.cm.gray)
                     # axes[1].imshow(labelslice,alpha=0.5,cmap=plt.cm.gray)
                     #

                     Slicelist.append(Batch)
                     labellist.append(labelslice)


        if axis == 'Sagittal':
            # self._Batch2D=self._array[:,:,0].ravel()
              for slice in arange(vol.shape[2]):
                Batch =vol[:, :,slice]
                labelslice=label[:,:,slice]

                if slice==0:
                   Slicelist.append(Batch)
                   labellist.append(labelslice)
                else:
                  corelation=pearsonr(array(Batch).ravel(),array(Slicelist)[len(Slicelist)-1].ravel())[0]
                  print('The patch correlation cofficient:%f'%corelation)
                  if corelation<0.9:
                     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
                     #
                     # axes[0].imshow(array(Slicelist)[len(Slicelist)-1],cmap=plt.cm.gray)
                     # axes[0].imshow(array(labellist)[len(labellist)-1],alpha=0.5,cmap=plt.cm.gray)
                     #
                     # axes[1].imshow(Batch,cmap=plt.cm.gray)
                     # axes[1].imshow(labelslice,alpha=0.5,cmap=plt.cm.gray)
                     #

                     Slicelist.append(Batch)
                     labellist.append(labelslice)

        if axis == 'Coronal':

              for slice in arange(vol.shape[1]):
                Batch =vol[:,slice, :]
                labelslice=label[:,slice,:]

                if slice==0:
                   Slicelist.append(Batch)
                   labellist.append(labelslice)
                else:
                  corelation=pearsonr(array(Batch).ravel(),array(Slicelist)[len(Slicelist)-1].ravel())[0]
                  print('The patch correlation cofficient:%f'%corelation)
                  if corelation<0.9:
                     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
                     #
                     # axes[0].imshow(array(Slicelist)[len(Slicelist)-1],cmap=plt.cm.gray)
                     # axes[0].imshow(array(labellist)[len(labellist)-1],alpha=0.5,cmap=plt.cm.gray)
                     #
                     # axes[1].imshow(Batch,cmap=plt.cm.gray)
                     # axes[1].imshow(labelslice,alpha=0.5,cmap=plt.cm.gray)
                     #

                     Slicelist.append(Batch)
                     labellist.append(labelslice)

        #self._Batch2D = array(Slicelist)
        return array(Slicelist),array(labellist)
#        print(self._Batch2D)

#*********Batch 2D image data from a series of patches which are extrated from volume data
    def Batch2DArray(self,label,direction='Axial'):
###Different slice direction data can be got and batch process
        Slicelist=[]
        if direction=='Axial':
            #self._Batch2D=self._array[0,:,:].ravel()

            for slice in arange(self._array.shape[0]):
                #if slice==0:
                #    continue
              #  else:
                    Batch=self._array[slice,:,:].ravel()
                    #self._Batch2D=vstack((self._Batch2D,Batch))
                    Slicelist.append(Batch)
           
            label=[label]*self._array.shape[0]
        if direction=='Sagittal':
            #self._Batch2D=self._array[:,:,0].ravel()
            for slice in arange(self._array.shape[2]):
             #   if slice==0:
               #     continue
             #   else:
                    Batch=self._array[:,:,slice].ravel()
              #      self._Batch2D=vstack((self._Batch2D,Batch))
                    Slicelist.append(Batch)
            label=[label]*self._array.shape[2]
        if direction=='Coronal':
           # self._Batch2D=self._array[:,0,:].ravel()
            for slice in arange(self._array.shape[1]):
            #    if slice==0:
            #        continue
            #    else:
                    Batch=self._array[:,slice,:].ravel()
                #    self._Batch2D=vstack((self._Batch2D,Batch))
                    Slicelist.append(Batch)
            label=[label]*self._array.shape[1]
        self._Batch2D=array(Slicelist)
        label=array(label)
        label.shape=(label.size,1)
        #print(label)
        self._Batch2D=hstack((label,self._Batch2D))
        #set_printoptions(threshold='nan') 
        print(self._Batch2D)
#*********Batch 3D image data from a series of patches which are extrated from CT,but make sure the input batch3D array have been flatten
    def Batch3DArray(self,label):   
        self._Batch3D=hstack((label,self._array.ravel()))  
        #self._Batch3D=vstack((self._Batch3D,Batch,batch3Darray))
    #convert Numpy Array to image
    def ConvertArray2PNG(self,array):
        self.PNGimg=scipy.misc.toimage(array)
