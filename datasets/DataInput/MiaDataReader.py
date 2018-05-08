from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import vtk
import os
import math
import sys
import xlrd
#import linecache
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import MiaUtils.Miaimshow as ms
import matplotlib.patches as patches
import SimpleITK as itk
from numpy import *
import random as rand
import tensorflow as tf
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.transform import resize as imresize
from datasets.DataInput import MiaDataPreprocess

tf.app.flags.DEFINE_string(
     'debug', False, 'The debug flag for display the images.')
tf.app.flags.DEFINE_string(
     'datadir', 'D:\DATA\\results', 'The file path for the images save.')
FLAGS = tf.app.flags.FLAGS
MIAPATCHSIZE=32
CLASSNAMES={
            0:'Others',
            1:'liver',
            2:'ambiguity'
            }
CLASSNAMESID={
            'Others':0,
            'liver':1,
            'ambiguity':2
            }
class MiaDataReader(object):
  def __init__(self,logger):
     # self.input=input
      self.output=[]
      self._3DVoxpatcharray=[]
      self._3DRegionpatcharray=[]
      self._3DSuperpatch=[]
      self._2Dpixelpatcharray=[]
      self.logger=logger


  def SetInputData(self,input):
      self.input=input
  def GetOutputData(self):
      return self.output
#return the input filename 
  def GetFilename(self):
      return self.input 
#return the Array data
  def GetArray(self):
      return self._numpy_array
  # ***Get2D each pixel patch array of label data
  def Get2DpixelpatchArray(self):
      return self._2Dpixelpatcharray
#***Get3D volume patch
  def Get3DVTKpatch(self):
      return self._3Dpatchoutput
#***Get3D volume patch
  def Get3DpatchArray(self):
      return self._3Dpatcharray
# ***Get3D each voxel patch array of label data
  def Get3DVoxpatchArray(self):
      return self._3DVoxpatcharray
# ***Get3D region patch array of label data
  def Get3DRegionpatchArray(self):
      return self._3DRegionpatcharray

  # ***Get3D superpixel-based region patch array of label data
  def Get3DSuperpixelpatchArray(self):
      return  self._3DSuperpatch
#return Axial slice
  def GetSliceAxial(self):
      return self.resliceAxial
#return Coronal slice
  def GetSliceCoronal(self):
      return self.resliceCoronal
#return Sagittal
  def GetSliceSagittal(self):
      return self.resliceSagittal
#return Obliques
  def GetSliceSagittal(self):
      return self.resliceOblique
#return size of data
  def Getsize(self):
      return self._MiaSize
#return spacing of data
  def Getspacing(self):
      return self._MiaSpacing
#return orgin of data
  def Getorgin(self):
      return self._MiaOrgin
#return dimension of data
  def Getdimension(self):
      return self._MiaDims
#return channel of data
  def Getchannel(self):
      return self._MiaChannel

  def update(self):
      FileType=os.path.splitext(self.input)[1].split('.')[-1]
      try:
        if FileType=='dcm':#no test
          reader = itk.ImageSeriesReader()
          seriesIDread=reader.GetGDCMSeriesIDs(self.input)[1]
          filenamesDICOM = reader.GetGDCMSeriesFileNames(self.input,seriesIDread)
          reader.SetFileNames(filenamesDICOM)
          self._numpy_array = reader.Execute()

        elif (FileType=='nrrd'or'nii'):
         datareader=itk.ImageFileReader()
         datareader.SetFileName(self.input)
         reader=datareader.Execute()
         #reader=itk.ReadImage(self.input)
         self._MiaDims=reader.GetDimension()
         self._MiaChannel=reader.GetNumberOfComponentsPerPixel()
        # print(MiaDims)
         self._MiaSize=reader.GetSize()
         print('Data size before convert to Array:[%d,%d,%d]'%(self._MiaSize[0],self._MiaSize[1],self._MiaSize[2]))
         self._MiaSpacing=reader.GetSpacing()
         print('Data spacing before convert to Array:[%f,%f,%f]'%(self._MiaSpacing[0],self._MiaSpacing[1],self._MiaSpacing[2]))
         self._MiaOrgin=reader.GetOrigin()
         print('Data origin before convert to Array:[%f,%f,%f]'%(self._MiaOrgin[0],self._MiaOrgin[1],self._MiaOrgin[2]))
         outputspacing=[1,1,1]#convert the voxel size into  1mm at all axises
         outputsize=int16(array(self._MiaSpacing)*array(self._MiaSize)/array(outputspacing)).tolist()

         resample=itk.ResampleImageFilter()
         resample.SetOutputOrigin(self._MiaOrgin)
         resample.SetOutputDirection(reader.GetDirection())

         resample.SetOutputSpacing(outputspacing)
         resample.SetSize(outputsize)


         reader=resample.Execute(reader)

         self._numpy_array=itk.GetArrayFromImage(reader)# convert to numpy Array from image
         print(self._numpy_array.shape)


        else:
           print('The input file is not found')
      except IOError as err:
        print('File Error:'+str(err))
  def Extract3DSuperPatch(self,imgarray,labelarray,classname,labelcount,
                          numsample=1000,numsuperpixels=150,stride=32,
                          oneslice=True,flagslice=True):
    """
    Extract each supervoxel patch based on slic supervoxel computing

    """
    escope=16
    Allimagelist=[]

    borderindex = self.BorderImage(labelarray.nonzero())

    vol=imgarray[max(borderindex[0],borderindex[0] - escope):min(borderindex[1],borderindex[1] + escope),
                max(borderindex[2],borderindex[2] - escope):min(borderindex[3],borderindex[3] + escope)
                                   :]
    #adjust the window-level to enhance organ region
    vol[vol>200]=200
    vol[vol<-200]=-200
    #self.VolSave('D:\DATA','liverwltest.nii',vol)

    label=labelarray[max(borderindex[0],borderindex[0] - escope):min(borderindex[1],borderindex[1] + escope),
                     max(borderindex[2],borderindex[2] - escope):min(borderindex[3],borderindex[3] + escope)
                     :]
   # self.VolSave('D:\DATA','liverlabeltest.nii',label)
    if flagslice:
      if oneslice:

          imgslice,labelslice,bordrer,_=self.SalientsliceExtract(vol,label,direction='Axial')
          imgslice=expand_dims(imgslice,axis=0)
          labelslice=expand_dims(labelslice,axis=0)
          if max(labelslice[0].ravel())!=len(CLASSNAMES):
              self.logger.info('The number of label %d not equal to numbers of classes %d'%(max(labelslice[0].ravel()),len(CLASSNAMES)))
          for i in range(imgslice.shape[0]):
           self.logger.info('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           print('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           #convert to range at 0-1
           factor = 0.00001
          # slicelist=[]
           PatchNorm = divide(((imgslice[i].ravel() - min(imgslice[i].ravel()))),
                              ((max(imgslice[i].ravel()) - min(imgslice[i].ravel())) + factor))

           PatchNorm= PatchNorm.reshape(imgslice[i].shape)

           from skimage.filters import gaussian
           PatchNorm = gaussian(PatchNorm, sigma=0.5)

           slice_colors = dstack((dstack((PatchNorm, PatchNorm)), PatchNorm))

           superpixel = slic(slice_colors, n_segments=numsuperpixels, compactness=5, sigma=1.0)#superpixel caculation wiht slic
           slice_entro = entropy(PatchNorm, disk(5))#get the whole image entropy
           region = regionprops(superpixel,slice_entro)#get the all regions information
      #******************************************************************************#
       #display the superpixel results
      #*******************************************************************************#
           if FLAGS.debug:
             fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
                                    subplot_kw={'adjustable': 'box-forced'})
             ax = axes.ravel()
             ax[0].imshow(slice_colors)
             from skimage.segmentation import mark_boundaries
             ax[1].imshow(mark_boundaries(slice_colors, superpixel))
             ax[1].imshow(labelslice[i], cmap='hot', alpha=0.5)
             ax[2].imshow(slice_entro)
             ax[3].imshow(superpixel)

           for classid in range(int(max(labelslice[0].ravel()))+1):

              if classid==0:
                  thval=0.9
                  imglist,edgepatchsize=self.SuperpixelPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)

                  if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)
                  if edgepatchsize!=0:
                     labelcount['ambiguity']+=edgepatchsize

              elif classid==2:
                  break

              else:
                 thval=0.5
                 imglist,edgepatchsize=self.SuperpixelPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)
                 if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)
                 if edgepatchsize!=0:
                     labelcount['ambiguity']+=edgepatchsize

      else:
       slice = MiaDataPreprocess.MiaDataPreprocess()
       # Get the data 2d slice
       imgslice,labelslice=slice.Slice2DArray_salient(vol,label,axis='Axial')
       for i in range(imgslice.shape[0]):
           self.logger.info('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           print('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           #convert to range at 0-1
           factor = 0.00001
          # slicelist=[]
           PatchNorm = divide(((imgslice[i].ravel() - min(imgslice[i].ravel()))),
                              ((max(imgslice[i].ravel()) - min(imgslice[i].ravel())) + factor))

           PatchNorm= PatchNorm.reshape(imgslice[i].shape)
           from skimage.filters import gaussian
           PatchNorm = gaussian(PatchNorm, sigma=0.5)
           slice_colors = dstack((dstack((PatchNorm, PatchNorm)), PatchNorm))

           superpixel = slic(slice_colors, n_segments=numsuperpixels, compactness=5, sigma=1.0)#superpixel caculation wiht slic
           slice_entro = entropy(PatchNorm, disk(5))#get the whole image entropy
           region = regionprops(superpixel,slice_entro)#get the all regions information
      #******************************************************************************#
       #display the superpixel results
      #*******************************************************************************#
           if FLAGS.debug:
             fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
                                    subplot_kw={'adjustable': 'box-forced'})
             ax = axes.ravel()
             ax[0].imshow(slice_colors)
             from skimage.segmentation import mark_boundaries
             ax[1].imshow(mark_boundaries(slice_colors, superpixel))
             ax[1].imshow(labelslice[i], cmap='hot', alpha=0.5)
             ax[2].imshow(slice_entro)
             ax[3].imshow(superpixel)
       #exclude the tumor
           for classid in range(int(max(labelslice[0].ravel()))+1):
              if classid==0:
                  thval=0.7
                  imglist,edgepatchsize=self.SuperpixelPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)
                  if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)
                  if edgepatchsize!=0:
                     labelcount['ambiguity']+=edgepatchsize

              elif classid==2:
                  break
              else:
                 thval=0.3
                 imglist,edgepatchsize=self.SuperpixelPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)
                 if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)
                 if edgepatchsize!=0:
                     labelcount['ambiguity']+=edgepatchsize

    else:
       pass

    self.logger.info('Extracted %d Patches with size is:[Height=%d,Width=%d]' %(len(imglist), stride, stride))


    return Allimagelist,edgepatchsize

  def SuperpixelPatch(self,region,superpixel,slice_entro,PatchNorm,labelslice,
                      thval,classid,classname,numsuperpixels=150,stride=32):
      """
      Extract the superpixel patches based on entropy-rate
      :param region:superpixel-region with entropy image information
      :param PatchNorm:narray input image after scale into 0-255
      :param labelslice:narray input label image
      :param thval:ratio of ROI determined which class of the superpixel
      :param classid:the class for patch extracting
      :param classname:the name of class to assign
      :param numsuperpixels:number of superpiexels
      :param stride:the size of patch
      :return: list of patches have been extracted
      """
     # region_index=[]
      patchcount=0
      edgepatchcount=0
    # patcharray=array([])
      imglist=[]
      stridecenter=int(stride/2)
      OrgantestCord=[]
      EdgetestCord=[]
      EdgeSurCord=[]

      for r in range(len(region)):

               if (region[r].mean_intensity)>1.5:  #if the superpixel region of entropy is less than threshold, the region won't be choosed for the training set
                  dataNorm={'image':array([]),'label':0}
                  if int(region[r].centroid[0])- stridecenter>0 and int(region[r].centroid[0])+stridecenter<PatchNorm.shape[0] and \
                         int(region[r].centroid[1]) - stridecenter > 0 and int(region[r].centroid[1]) + stridecenter <PatchNorm.shape[1]:
                # if the region of superpixel has more than 90% organ label, it will be assigned as organ class
                    flagclass=(labelslice[int(region[r].centroid[0]) - stridecenter:int(region[r].centroid[0]) + stridecenter,
                                         int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter] ==classid).sum() / \
                                        labelslice[int(region[r].centroid[0]) - stridecenter:int(region[r].centroid[0]) + stridecenter,
                                                      int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter].size
                  #assign different class according to the label pixel ratio of superpixle
                    if flagclass==1:
                      #region_index.append(region[r].centroid) #test for superpixel to record the region centroid index

                          dataNorm['image'] = self.DataNormalize(PatchNorm[int(region[r].centroid[0])- stridecenter:int(region[r].centroid[0]) + stridecenter,
                                                       int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter])
                          #ms.imshow(dataNorm['image'],num=patchcount+10)
                          if (dataNorm['image'] == 0).all():
                             pass
                          else:
                             dataNorm['label']=classid
                             imglist.append(dataNorm)
                             OrgantestCord.append(array([int(region[r].centroid[0]),int(region[r].centroid[1])]))
                             patchcount+=1
                             self.logger.info("%dth superpixel patches have been extracted "%patchcount)
                             print("%dth superpixel patches have been extracted "%patchcount)
                  #for prosessing the edge of organ

                    elif (flagclass>=thval and flagclass<1):
                        # Center of superpixel and surrounding 4 pixel patch will be extracted
                          #***************************************************
                          ###extract the  center of superpixel region patch
                          #***********************************************************

                          dataNorm['image'] = self.DataNormalize(PatchNorm[int(region[r].centroid[0])- stridecenter:int(region[r].centroid[0]) + stridecenter,
                                                       int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter])
                          #ms.imshow(dataNorm['image'],num=patchcount+10)
                          if (dataNorm['image'] == 0).all():
                             pass
                          else:
                             if classid==0:
                                dataNorm['label']=classid
                                patchcount+=1
                             else:
                                 #This will be deprecated in future, here just add more patches at border for trainning ,we donot need add an extra class
                                dataNorm['label']=CLASSNAMESID['ambiguity']
                                edgepatchcount+=1

                             imglist.append(dataNorm)
                             EdgetestCord.append(array([int(region[r].centroid[0]),int(region[r].centroid[1])]))

                             self.logger.info("Edge:%dth superpixel patches have been extracted "%edgepatchcount)
                             print("Edge:%dth superpixel patches have been extracted "%edgepatchcount)

                          #***************************************************
                          ###extract the 4 neighbor around the center of superpixel region patch
                          #***********************************************************
                          if classid!=0:
                            min_y, min_x, max_y, max_x=region[r].bbox
                            SPsurroundxy=[]
                            y1=int(region[r].centroid[0]-(region[r].centroid[0]-min_y)/2)
                            x1=int(region[r].centroid[1])
                            SPsurroundxy.append(array([y1,x1]))

                            y2=int(region[r].centroid[0])
                            x2=int(region[r].centroid[1]+(max_x-region[r].centroid[1])/2)
                            SPsurroundxy.append(array([y2,x2]))
                            y3=int(region[r].centroid[0]+(max_y-region[r].centroid[0])/2)
                            x3=int(region[r].centroid[1])
                            SPsurroundxy.append(array([y3,x3]))

                            y4=int(region[r].centroid[0])
                            x4=int(region[r].centroid[1]-(region[r].centroid[1]-min_x)/2)
                            SPsurroundxy.append(array([y4,x4]))

                            def patchextract(PatchNorm,labelslice,y,x,stridecenter,thval):
                                datapatch=array([])
                                label=None
                                if y-stridecenter>0 and y+stridecenter<PatchNorm.shape[0] and \
                                    x-stridecenter>0 and x+stridecenter<PatchNorm.shape[1]:

                                    flagclass=(labelslice[y-stridecenter :y+stridecenter,x-stridecenter:x+stridecenter] == classid).sum() / \
                                        labelslice[y-stridecenter :y+stridecenter,x-stridecenter:x+stridecenter].size
                                    if flagclass==1:
                                         datapatch=self.DataNormalize(PatchNorm[y-stridecenter :y+stridecenter,x-stridecenter:x+stridecenter])
                                         label=1
                                    if flagclass>=thval and flagclass<1:
                                        datapatch=self.DataNormalize(PatchNorm[y-stridecenter :y+stridecenter,x-stridecenter:x+stridecenter])
                                        label=2
                  #assign different class according to the label pixel ratio of superpixle


                                return datapatch,label


                            for sp in SPsurroundxy:
                               dataNorm={'image':array([]),'label':0}
                               patchsur,labelmark=patchextract(PatchNorm,labelslice,sp[0],sp[1],stridecenter,thval)
                               if patchsur.any():
                                 if (patchsur ==0).all():
                                     pass
                                 else:
                                   dataNorm['image']=patchsur
                                   dataNorm['label']=labelmark
                                   imglist.append(dataNorm)
                                   if labelmark==2:
                                     edgepatchcount+=1
                                     EdgeSurCord.append(array(sp))
                                   elif labelmark==1:
                                     patchcount+=1
                                     OrgantestCord.append(array(sp))

###### SuperPatchshow
           # from skimage.segmentation import mark_boundaries
           # fig = plt.figure(figsize=(9, 5))
           # ax = fig.add_subplot(111)
           # plt.gray()
           # ax.imshow(mark_boundaries(PatchNorm, superpixel))
           # figcolor=ax.imshow(slice_entro, cmap='rainbow',alpha=0.5)
           # ax.imshow(labelslice[i], cmap='hot', alpha=0.5)
           # fig.colorbar(figcolor, ax=ax)
           # #ax.imshow(PatchNorm)
           # ax.plot(transpose(array(OrgantestCord))[1],transpose(array(OrgantestCord))[0],'yo',lw=15)
           # for c in range(len(OrgantestCord)):
           #     ax.add_patch(patches.Rectangle((OrgantestCord[c][1]-16, OrgantestCord[c][0]-16),# (x,y)
           #                                              32,          # width
           #                                              32,          # heigh
           #                                              fill=False,lw=5,edgecolor='b'
           #                                              ))
           # ax.axis('off')
           # plt.show()
      return imglist,edgepatchcount
  def Extract3DSuperNoborderPatch(self,imgarray,labelarray,classname,labelcount,
                          numsample=1000,numsuperpixels=150,stride=32,
                          oneslice=True,flagslice=True):
    """

    """
    escope=0
    Allimagelist=[]


    borderindex = self.BorderImage(labelarray.nonzero())

    vol=imgarray[borderindex[0] - escope:borderindex[1] + escope,
                                    :,
                                    :]
    #adjust the window-level to enhance organ region
    vol[vol>200]=200
    vol[vol<-200]=-200
    #self.VolSave('D:\DATA','liverwltest.nii',vol)

    label=labelarray[borderindex[0] - escope:borderindex[1] + escope,
                      :,
                      :]
   # self.VolSave('D:\DATA','liverlabeltest.nii',label)
    if flagslice:
      if oneslice:
          imgslice,labelslice,bordrer,_=self.SalientsliceExtract(vol,label)
          imgslice=expand_dims(imgslice,axis=0)
          labelslice=expand_dims(labelslice,axis=0)
          if max(labelslice[0].ravel())!=len(CLASSNAMES):
              self.logger.info('The number of label %d not equal to numbers of classes %d'%(max(labelslice[0].ravel()),len(CLASSNAMES)))
          for i in range(imgslice.shape[0]):
           self.logger.info('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           print('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           #convert to range at 0-1
           factor = 0.00001
          # slicelist=[]
           PatchNorm = divide(((imgslice[i].ravel() - min(imgslice[i].ravel()))),
                              ((max(imgslice[i].ravel()) - min(imgslice[i].ravel())) + factor))

           PatchNorm= PatchNorm.reshape(imgslice[i].shape)

           from skimage.filters import gaussian
           PatchNorm = gaussian(PatchNorm, sigma=0.5)

           slice_colors = dstack((dstack((PatchNorm, PatchNorm)), PatchNorm))

           superpixel = slic(slice_colors, n_segments=numsuperpixels, compactness=5, sigma=1.0)#superpixel caculation wiht slic
           slice_entro = entropy(PatchNorm, disk(5))#get the whole image entropy
           region = regionprops(superpixel,slice_entro)#get the all regions information
      #******************************************************************************#
       #display the superpixel results
      #*******************************************************************************#
           if FLAGS.debug:
             fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
                                    subplot_kw={'adjustable': 'box-forced'})
             ax = axes.ravel()
             ax[0].imshow(slice_colors)
             from skimage.segmentation import mark_boundaries
             ax[1].imshow(mark_boundaries(slice_colors, superpixel))
             ax[1].imshow(labelslice[i], cmap='hot', alpha=0.5)
             ax[2].imshow(slice_entro)
             ax[3].imshow(superpixel)

           for classid in range(int(max(labelslice[0].ravel()))+1):

              if classid==0:
                  thval=0.9
                  imglist=self.SuperpixelNoborderPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)

                  if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)

              elif classid==2:
                  break

              else:
                 thval=0.5
                 imglist=self.SuperpixelNoborderPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)
                 if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)

      else:
       slice = MiaDataPreprocess.MiaDataPreprocess()
       # Get the data 2d slice
       imgslice,labelslice=slice.Slice2DArray_salient(vol,label,axis='Axial')
       for i in range(imgslice.shape[0]):
           self.logger.info('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           print('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           #convert to range at 0-1
           factor = 0.00001
          # slicelist=[]
           PatchNorm = divide(((imgslice[i].ravel() - min(imgslice[i].ravel()))),
                              ((max(imgslice[i].ravel()) - min(imgslice[i].ravel())) + factor))

           PatchNorm= PatchNorm.reshape(imgslice[i].shape)
           # fig,axes=plt.subplots(1,2)
           # axes[0].imshow(PatchNorm,cmap=plt.cm.gray)
           # axes[0].set_title('before gaussian filter')

           from skimage.filters import gaussian
           PatchNorm = gaussian(PatchNorm, sigma=0.5)

           # axes[1].imshow(PatchNorm,cmap=plt.cm.gray)
           # axes[1].set_title('after gaussian filter')

           slice_colors = dstack((dstack((PatchNorm, PatchNorm)), PatchNorm))

           superpixel = slic(slice_colors, n_segments=numsuperpixels, compactness=5, sigma=1.0)#superpixel caculation wiht slic
           slice_entro = entropy(PatchNorm, disk(5))#get the whole image entropy
           region = regionprops(superpixel,slice_entro)#get the all regions information
      #******************************************************************************#
       #display the superpixel results
      #*******************************************************************************#

           for classid in range(int(max(labelslice[i].ravel()))+1):
              if classid==0:
                  thval=0.9
                  imglist=self.SuperpixelNoborderPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)
                  if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)

              elif classid==2:
                  break
              else:
                 thval=0.5
                 imglist=self.SuperpixelNoborderPatch(region,superpixel,slice_entro,PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsuperpixels=numsuperpixels,stride=stride)
                 if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)

    else:
        thval=0.9
        imglist=self.SuperVoxPatch(vol,label,thval,classname,
                               numsuperpixels=numsuperpixels,stride=stride)

    self.logger.info('Extracted %d Patches with size is:[Height=%d,Width=%d]' %(len(imglist), stride, stride))


    return Allimagelist

  def SuperpixelNoborderPatch(self,region,superpixel,slice_entro,PatchNorm,labelslice,
                      thval,classid,classname,numsuperpixels=150,stride=32):
      """
      Extract the superpixel patches based on entropy-rate
      :param region:superpixel-region with entropy image information
      :param PatchNorm:narray input image after scale into 0-255
      :param labelslice:narray input label image
      :param thval:ratio of ROI determined which class of the superpixel
      :param classid:the class for patch extracting
      :param classname:the name of class to assign
      :param numsuperpixels:number of superpiexels
      :param stride:the size of patch
      :return: list of patches have been extracted
      """
     # region_index=[]
      patchcount=0
      edgepatchcount=0
    # patcharray=array([])
      imglist=[]
      stridecenter=int(stride/2)
      OrgantestCord=[]
      EdgetestCord=[]
      EdgeSurCord=[]

      for r in range(len(region)):

               if (region[r].mean_intensity)>1.5:  #if the superpixel region of entropy is less than threshold, the region won't be choosed for the training set
                  dataNorm={'image':array([]),'label':0}
                  if int(region[r].centroid[0])- stridecenter>0 and int(region[r].centroid[0])+stridecenter<PatchNorm.shape[0] and \
                         int(region[r].centroid[1]) - stridecenter > 0 and int(region[r].centroid[1]) + stridecenter <PatchNorm.shape[1]:
                # if the region of superpixel has more than 90% organ label, it will be assigned as organ class
                    flagclass=(labelslice[int(region[r].centroid[0]) - stridecenter:int(region[r].centroid[0]) + stridecenter,
                                         int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter] ==classid).sum() / \
                                        labelslice[int(region[r].centroid[0]) - stridecenter:int(region[r].centroid[0]) + stridecenter,
                                                      int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter].size
                  #assign different class according to the label pixel ratio of superpixle
                    if (flagclass>=thval and flagclass<=1):
                      #region_index.append(region[r].centroid) #test for superpixel to record the region centroid index

                          dataNorm['image'] = self.DataNormalize(PatchNorm[int(region[r].centroid[0])- stridecenter:int(region[r].centroid[0]) + stridecenter,
                                                       int(region[r].centroid[1]) - stridecenter:int(region[r].centroid[1]) + stridecenter])
                          #ms.imshow(dataNorm['image'],num=patchcount+10)
                          if (dataNorm['image'] == 0).all():
                             pass
                          else:
                             dataNorm['label']=classid
                             imglist.append(dataNorm)
                             OrgantestCord.append(array([int(region[r].centroid[0]),int(region[r].centroid[1])]))
                             patchcount+=1
                             self.logger.info("%dth superpixel patches have been extracted "%patchcount)
                             print("%dth superpixel patches have been extracted "%patchcount)
                #for prosessing the edge of organ
      #show the patch position located at orginal image

      return imglist

      # ******Extract whole volume data with whole groundtruth label information
  def Parsingimages(self,img,label,shape):
      """

      :param img: ndarray image
      :param label: ndarray label data
      :param shape: the shape will be resized
      :return:data pair dictionary with image and label
      """
      datapairs={'image':array([]),'label':array([])}
      datalist=[]
      img=imresize(img,(shape[0],shape[1]),mode='reflect',preserve_range=True)
      label=imresize(label,(shape[0],shape[1]),mode='reflect',preserve_range=True)
      datapairs['image']=img
      datapairs['label']=label
      datalist.append(datapairs)
      return datalist

  def SalientsliceExtract(self,vol,vollabel,direction='Axial'):
      """
      Input:
      vol:The 3D volume data
      volabel:vole labeled data

      :return:
      salient slice image
      salient slice label image
      border of image for computing
      """
      from collections import Counter
      Labelindex = vollabel.nonzero()  # get the nonzero indices
      print('There are %d voxels in label data'%(len(Labelindex[0])))
      if direction=='Axial':
         cor=Counter(list(Labelindex[0]))# count the number of label pixel at dims=0
         if cor.most_common(1)[0][0]:
            img=vol[cor.most_common(1)[0][0]]#get the slice with most  number of label pixel
            label=vollabel[cor.most_common(1)[0][0]]
      elif direction=='Sagittal':
         cor=Counter(list(Labelindex[2]))# count the number of label pixel at dims=0
         if cor.most_common(1)[0][0]:
            img=vol[:,:,cor.most_common(1)[0][0]]#get the slice with most  number of label pixel
            label=vollabel[:,:,cor.most_common(1)[0][0]]
      elif direction=='Coronal':
         cor=Counter(list(Labelindex[1]))# count the number of label pixel at dims=0
         if cor.most_common(1)[0][0]:
            img=vol[:,cor.most_common(1)[0][0],:]#get the slice with most  number of label pixel
            label=vollabel[:,cor.most_common(1)[0][0],:]
      borderindex = self.BorderImage(label.nonzero())
      salientindex=cor.most_common(1)[0][0]
      return img,label,borderindex,salientindex

  #******Extract each pixel's patches with stride*stride size and random sample
  def Extract2DPixelPatch(self, volarray, vollabel,labelcount,numSample=1000,stride=32):
       """
       #in order to speed up search, so avoid to traverse all of voxels
       select the most numbers of slice
       """
       Allimagelist=[]
       slice = MiaDataPreprocess.MiaDataPreprocess()
       # adjust the window-level to enhance organ region
       volarray[volarray > 200] = 200
       volarray[volarray < -200] = -200
       # Get the data 2d slice
       imgslice,labelslice=slice.Slice2DArray_salient(volarray,vollabel,axis='Axial')


       for i in range(imgslice.shape[0]):
           self.logger.info('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           print('%dth/%d imageslice is being extracted patch'%(i,imgslice.shape[0]))
           #convert to range at 0-1
           factor = 0.00001
          # slicelist=[]
           PatchNorm = divide(((imgslice[i].ravel() - min(imgslice[i].ravel()))),
                              ((max(imgslice[i].ravel()) - min(imgslice[i].ravel())) + factor))

           PatchNorm= PatchNorm.reshape(imgslice[i].shape)
           if FLAGS.debug:
               fig,axes=plt.subplots(1,2)
               axes[0].imshow(PatchNorm,cmap=plt.cm.gray)
               axes[0].set_title('before gaussian filter')

           from skimage.filters import gaussian
           PatchNorm = gaussian(PatchNorm, sigma=0.5)

           #exclude the tumor
           for classid in range(int(max(labelslice[i].ravel()))+1):
              if classid==0:
                  thval=0.9
                  imglist=self.PixelNoborderPatch(PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                  numsample=numSample,stride=stride)
                  if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)
                  # if edgepatchsize!=0:
                  #    labelcount['ambiguity']+=edgepatchsize

              elif classid==2:
                  break
              else:
                 thval=0.1
                 imglist=self.PixelNoborderPatch(PatchNorm,labelslice[i],thval,classid,CLASSNAMES[classid],
                                                               numsample=numSample,stride=stride)
                 if len(imglist) != 0:
                     labelcount[CLASSNAMES[classid]]+=len(imglist)
                     Allimagelist.extend(imglist)

       return Allimagelist

  def PixelNoborderPatch(self,PatchNorm,labelslice,
                      thval,classid,classname,numsample=1000,stride=32):
      """
      Extract the superpixel patches based on entropy-rate
      :param region:superpixel-region with entropy image information
      :param PatchNorm:narray input image after scale into 0-255
      :param labelslice:narray input label image
      :param thval:ratio of ROI determined which class of the superpixel
      :param classid:the class for patch extracting
      :param classname:the name of class to assign
      :param numsuperpixels:number of superpiexels
      :param stride:the size of patch
      :return: list of patches have been extracted
      """
     # region_index=[]
      patchcount=0

    # patcharray=array([])
      imglist=[]
      stridecenter=int(stride/2)
      escope=stride*2

      Labelindex=array([])
      Labelindex=(labelslice==classid).nonzero()
      numSample=min((numsample,len(Labelindex[0])))# set numbers of sample as max between numSample and data size, as some organ size is small than seting
      Randlabelindex=rand.sample(list(transpose(Labelindex)),numSample)

      pixelcount=0
      imglist=[]
      TestCord=[]
      stridecenter=int(stride/2)
      OutRangeFlag=False
      for j in range(len(Randlabelindex)):
          #Make sure that the all label voxel have a stride*stride*stride patch,not out of range in imagearray
                  dataNorm={'image':array([]),'label':0}
                  if int(Randlabelindex[j][0])- stridecenter>0 and int(Randlabelindex[j][0])+stridecenter<PatchNorm.shape[0] and \
                         int(Randlabelindex[j][1]) - stridecenter > 0 and int(Randlabelindex[j][1]) + stridecenter <PatchNorm.shape[1]:
                # if the region of superpixel has more than 90% organ label, it will be assigned as organ class
                    flagclass=(labelslice[int(Randlabelindex[j][0]) - stridecenter:int(Randlabelindex[j][0]) + stridecenter,
                                         int(Randlabelindex[j][1]) - stridecenter:int(Randlabelindex[j][1]) + stridecenter] ==classid).sum() / \
                                        labelslice[int(Randlabelindex[j][0]) - stridecenter:int(Randlabelindex[j][0]) + stridecenter,
                                                      int(Randlabelindex[j][1]) - stridecenter:int(Randlabelindex[j][1]) + stridecenter].size
                  #assign different class according to the label pixel ratio of superpixle
                    if (flagclass>=thval and flagclass<=1):

                          dataNorm['image'] = self.DataNormalize(PatchNorm[int(Randlabelindex[j][0])- stridecenter:int(Randlabelindex[j][0]) + stridecenter,
                                                       int(Randlabelindex[j][1]) - stridecenter:int(Randlabelindex[j][1]) + stridecenter])
                          #ms.imshow(dataNorm['image'],num=patchcount+10)
                          if (dataNorm['image'] == 0).all():
                             pass
                          else:
                             dataNorm['label']=classid
                             imglist.append(dataNorm)
                             TestCord.append(Randlabelindex[j])
                             patchcount+=1
                             self.logger.info("%dth pixel patches have been extracted "%patchcount)
                             print("%dth pixel patches have been extracted "%patchcount)

      if (len(Randlabelindex))>patchcount:
          print('The number of extracted pixel patch:%d are less than ground-truth label pixels:%d'%(patchcount,len(Randlabelindex)))

      #show the patch position located at orginal image
      if FLAGS.debug:
        imgtest=[]
        if len(imglist)>25:
             for tmpimg in imglist:
               imgtest.append(tmpimg['image'])
             ms.subplots(imgtest[0:25],num=100,cols=5)

        if len(TestCord)>5:
              # import MiaSeg.MiaUtils.Miaimshow as ms
               ms.PixelPatchshow(TestCord,PatchNorm,labelslice,color='b',stride=stride)


      return imglist

   ##****Extract the ROI 3Dvolume patches with groundtruth labels
  def Extract3DVolumePatch(self,imgarray,labelarray):
       #fig=plt.figure()
       #ax=plt.subplot(1,2,1)
       #ax.imshow(imgarray[70,:,:],cmap=cm.gray)
       #ax=plt.subplot(1,2,2)
       #ax.imshow(labelarray[70,:,:],cmap=cm.gray)
       #plt.show()
       Labelindex=labelarray.nonzero() #get the nonzero indices
       [zmaxid,ymaxid,xmaxid]=nanargmax(Labelindex,axis=1)
       [zminid,yminid,xminid]=nanargmin(Labelindex,axis=1)
       zmax=Labelindex[0][zmaxid]
       zmin=Labelindex[0][zminid]
       ymax=Labelindex[1][ymaxid]
       ymin=Labelindex[1][yminid]
       xmax=Labelindex[2][xmaxid]
       xmin=Labelindex[2][xminid]
 #spacing of imagedata equal to labelimage
#Make sure the patch size is 32*32*32*********
       [zmin,zmax,signflagz]=self.PatchFill(zmin,zmax,imgarray.shape[0])
       [ymin,ymax,signflagy]=self.PatchFill(ymin,ymax,imgarray.shape[1])
       [xmin,xmax,signflagx]=self.PatchFill(xmin,xmax,imgarray.shape[2])
       print('The label patch bounds:[z=(%d,%d),y=(%d,%d),x=(%d,%d)]'%(zmin,zmax,ymin,ymax,xmin,xmax))
       if signflagz or signflagy or signflagx:
          sampleInterz=(zmax-zmin)/MIAPATCHSIZE
          sampleIntery=(ymax-ymin)/MIAPATCHSIZE
          sampleInterx=(xmax-xmin)/MIAPATCHSIZE
          print('The sample interval at each axis=[z=%d,y=%d,x=%d]'%(sampleInterz,sampleIntery,sampleInterx))
          self._3Dpatcharray=imgarray[zmin:zmax:sampleInterz,ymin:ymax:sampleIntery,xmin:xmax:sampleInterx]#### if the patch up than difined size eg.32*32*32, smaple patch
       else:
          self._3Dpatcharray=imgarray[zmin:zmax,ymin:ymax,xmin:xmax]
       #plt.imshow(self._3Dpatch[10,:,:],cmap=cm.gray)
       #plt.show()
      #In order to computation efficiency, data need to be normalized
       # Minpixel=min(self._3Dpatcharray.ravel())
       # Maxpixel=max(self._3Dpatcharray.ravel())
       # PatchNorm=divide(((self._3Dpatcharray.ravel()-Minpixel)*255),(Maxpixel-Minpixel))
       # self._3Dpatcharray=PatchNorm.reshape(self._3Dpatcharray.shape)
       self._3Dpatcharray=self.DataNormalize(self._3Dpatcharray)
      #because the extraction patch is not C-contiguous, so first we need convert to C-Contiguous
       self._3Dpatcharray=self._3Dpatcharray.copy(order='C')
       self._3Dpatchoutput=self.miaNumpy2vtk(self._3Dpatcharray)
       print('Extracted Patch size is:[Depth=%d,Height=%d,Width=%d]'%(self._3Dpatcharray.shape[0],self._3Dpatcharray.shape[1],self._3Dpatcharray.shape[2]))

      # for data normalized caculation
  def DataNormalize(self,dataarray):
      #input narray
      #return normalize data with max and min into range[0,1]
      Minpixel = min(dataarray.ravel())
      Maxpixel = max(dataarray.ravel())
      factor=0.00001
      PatchNorm = divide(((dataarray.ravel() - Minpixel)*255), ((Maxpixel - Minpixel)+factor))
      return  PatchNorm.reshape(dataarray.shape)

  #Get any image border with known the location of nonzero region
  def BorderImage(self,borderindex):
      #input require must be position index
      if len(borderindex)==3:
        [zmaxid, ymaxid, xmaxid] = nanargmax(borderindex, axis=1)
        [zminid, yminid, xminid] = nanargmin(borderindex, axis=1)
        zmax = borderindex[0][zmaxid]
        zmin = borderindex[0][zminid]
        ymax = borderindex[1][ymaxid]
        ymin = borderindex[1][yminid]
        xmax = borderindex[2][xmaxid]
        xmin = borderindex[2][xminid]
        return [zmin,zmax,ymin,ymax,xmin,xmax]
      elif len(borderindex)==2:
          [ymaxid, xmaxid] = nanargmax(borderindex, axis=1)
          [yminid, xminid] = nanargmin(borderindex, axis=1)

          ymax = borderindex[0][ymaxid]
          ymin = borderindex[0][yminid]
          xmax = borderindex[1][xmaxid]
          xmin = borderindex[1][xminid]
          return [ymin, ymax, xmin, xmax]
      else:
          raise ValueError('Borderimage dims %d are not correct.'%len(borderindex))

#******when patch size is small than 32*32*32, extend the patch size ************
  def PatchFill(self,bmin,bmax,Datasize):
      sub=bmax-bmin
      if sub<=MIAPATCHSIZE:
         if bmax+(MIAPATCHSIZE-sub)<=Datasize:
            bmax=bmax+(MIAPATCHSIZE-sub)
         elif bmin-(MIAPATCHSIZE-sub)>=0:
              bmin=bmin-(MIAPATCHSIZE-sub)
         elif (bmax + int((MIAPATCHSIZE-sub)/2))<=Datasize and (bmin - int((MIAPATCHSIZE-sub)/2)>=0):
              bmax=bmax + int((MIAPATCHSIZE - sub) / 2)
              bmin=bmin - int((MIAPATCHSIZE - sub) / 2)
         else:
              # print('The patch size is too small to fail extration')
              raise ValueError('The patch size is too small to fail extration')
         signflag=False
      elif sub>MIAPATCHSIZE:
           subinter=self.Power2(sub)
           #sampleInter=sub/MIAPATCHSIZE
           if bmin+(sub-subinter)>=0:
              bmin=bmin+(sub-subinter)
           elif bmax-(sub-subinter)<=Datasize:
                bmax=bmax-(sub-subinter)
           else:
                #print('The patch size is too small to fail extration')
                raise ValueError('The patch size is too small to fail extration')
           signflag=True
      return [bmin,bmax,signflag]

#### if the patch up than difined size eg.32*32*32, smaple patch to make sure the suitable size
#  def SamplePatch(self,samplearray,sampinter):
#      Sample=samplearray[]

###convert the size into 2 power
  def Power2(self,value):
      if value<0:
         return 1
      else:
         i=0
         t=1
         t=math.pow(2,i)
         while t <= value:
               i+=1
               t=math.pow(2,i)
      return t
# Get the ROI
  def GetROI(self,ROI):
      voi = vtk.vtkExtractVOI()
      voi.SetInputConnection(self.output.GetOutputPort())
      voi.SetVOI(ROI)
      voi.SetSampleRate(1,1,1)
         #voi.SetSampleRate(3,3,3)
      voi.Update()#necessary for GetScalarRange()
      srange= voi.GetOutput().GetScalarRange()#needs Update() before!
      return voi
  #def _record(self):
  #    record = bytes(bytearray([self.label] + [self.output]))
  #    return record
#******Get the any angel slice from 3D volume***********
  def SetImageSlice(self):
   #***** Calculate the center of the volume*********
      (xMin, xMax, yMin, yMax, zMin, zMax) = self.output.GetOutput().GetExtent()
      (xSpacing, ySpacing, zSpacing) = self.output.GetOutput().GetSpacing()
      (x0, y0, z0) = self.output.GetOutput().GetOrigin()
      center = [x0 + xSpacing * 0.5 * (xMin + xMax),
                y0 + ySpacing * 0.5 * (yMin + yMax),
                z0 + zSpacing * 0.5 * (zMin + zMax)]
# Matrices for axial, coronal, sagittal, oblique view orientations
      axial = vtk.vtkMatrix4x4()#/axial slice  xoy plane
      axial.DeepCopy((1, 0, 0, center[0],
                      0, 1, 0, center[1],
                      0, 0, 1, center[2],
                      0, 0, 0, 1))

      coronal = vtk.vtkMatrix4x4()  #coronal slice xoz plane
      coronal.DeepCopy((1, 0, 0, center[0],
                        0, 0, 1, center[1],
                        0,-1, 0, center[2],
                        0, 0, 0, 1))

      sagittal = vtk.vtkMatrix4x4()  #sagittal slice yoz plane
      sagittal.DeepCopy((0, 0,-1, center[0],
                         1, 0, 0, center[1],
                         0,-1, 0, center[2],
                         0, 0, 0, 1))

      oblique = vtk.vtkMatrix4x4()  #oblique slice
      oblique.DeepCopy((1, 0, 0, center[0],
                        0, 0.866025, -0.5, center[1],
                        0, 0.5, 0.866025, center[2],
                        0, 0, 0, 1))
  # Extract a slice in the desired orientation
     
      self.resliceAxial=self.relice(axial)
      self.resliceCoronal=self.relice(coronal)
      self.resliceSagittal=self.relice(sagittal)
      self.resliceOblique=self.relice(oblique)
      
 #*******reslice using vtkImageReslice according to different transoform Mat
  def relice(self,vtkMat):
      slice=vtk.vtkImageReslice()
      slice.SetInputConnection(self.output.GetOutputPort())
      slice.SetOutputDimensionality(2)
      slice.SetResliceAxes(vtkMat)
      slice.SetInterpolationModeToCubic()
      slice.Update()
      return slice
###obtain the data file list
  def GetFilelist(self,filedir):
      filelist=[]
      filecount=0
      with open(filedir,'r') as f:
           #lines=f.readline()
           for line in f.readlines():
               line=line.strip()
               filelist.append(line)
               print(line)
               filecount+=1
      print(filecount)
      return filelist
  def xlsReader(self,filedir,sheetname):
      """
      This is xls files read
      :param filedir: The file path and name will be read
      :return:list of files
      """
      filelist=[]

      data=xlrd.open_workbook(filedir)
      tabel=data.sheet_by_name(sheetname)
      #filelist=tabel.col_slice(1,1,tabel.nrows)
      filelist=tabel.col_values(1)[1:tabel.nrows]
      print('there are %d files will be read'%tabel.nrows)
      return filelist
  def DataSave(self,filedir,fname,lname,type='2D'):
      #contents = b"".join([record for record, _ in records]
      if type=='2D':
         for slice in arange(self._3Dpatcharray.shape[0]):
             name=os.path.split(fname)[0].split('\\')[-1]+os.path.splitext(os.path.basename(lname))[0]+str(slice)+'.npy'
             filename=os.path.join(filedir,name)
             save(filename,self._3Dpatcharray[slice,:,:])
      elif type=='3D':
           name=os.path.splitext(os.path.basename(fname))[0]+os.path.splitext(os.path.basename(lname))[0]+'.npy'
           filename=os.path.join(filedir,name)
           save(filename,self._3Dpatcharray)
#******************Batch data save*********************************************
  def BatchSave(self,filedir,fname,BatchArray):
      
      filepath=os.path.join(filedir,fname)
      save(filepath,BatchArray)
     # filename=os.path.join(filedir,"Mia")
     #pen(filename, "wb").write(contents)
#******************information data save*********************************************
  def InfoSaveTxt(self,filedir,info):
      try:
         f=open(filedir,'w')
         for i in range(0,len(info)):
            #print(info,file=f)
             f.write(info[i]+'\n')
      except IOError as err:
         print('File Error:'+str(err))
      finally:
         f.close()
      #return True
#Nrrd 3d format data write
  def VolSave(self, filedir, fname, VolData):
      filepath = os.path.join(filedir, fname)
      itk.WriteImage(itk.GetImageFromArray(VolData),filepath)

      
  



