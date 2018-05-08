from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import xlwt
import xlsxwriter
import logging
import cv2
import vtk
import SimpleITK as itk
from PIL import Image
#from scipy.misc import imsave
from matplotlib.image import imsave
class MiaDataWriter(object):
    """
    This class for text or images or volume data writer into a specific file
    """
    def __init__(self):
       pass
    def xlsWriter(self,filedir,filename,sheetname,data):
        """
        writer the input data into a xls file with given filedir and file name
        :param filedir:The path for file where will be stored
        :param filename:The file name
        :param data:narray
        :return:bool type True when data have been successfully stored
        """
        workbook = xlwt.Workbook(encoding = 'ascii')
        worksheet = workbook.add_sheet(sheetname)
        dindex=0
        worksheet.write(dindex,0,'No.')
        worksheet.write(dindex,1,'file')
        for d in data.tolist():
            dindex+=1
            worksheet.write(dindex,0,dindex)
            worksheet.write(dindex,1,d)
        try:
           workbook.save(os.path.join(filedir,filename))
        except IOError as err:
            print('Data write error:'+str(err))
            return False
        else:
            print('Successfully finished writing')
            return True

    def xlsndarrayWriter(self,filedir,filename,data):
        """
        writer the input data into a xls file with given filedir and file name
        :param filedir:The path for file where will be stored
        :param filename:The file name
        :param data:list of dictinary with narray and items name in line with sheetname
        :return:bool type True when data have been successfully stored
        """

        workbook = xlsxwriter.Workbook(os.path.join(filedir,filename))
        worksheet=workbook.add_worksheet()
        col=0
        row=0
        try:
         for dataitem,value in data.items():

             if len(value.shape)==3:
               col=0
               worksheet=workbook.add_worksheet(dataitem)
               for dataarray in value.tolist():
                   for row,datawrite in enumerate(dataarray):
                    worksheet.write_row(row, col, datawrite)
             elif len(value.shape)==2:
                worksheet=workbook.add_worksheet(dataitem)
                row=0
                for col,datawrite in enumerate(value.tolist()):
                    worksheet.write_column(row, col, datawrite)

             elif len(value.shape)==1:
                  row=0
                  worksheet.write(row, col, dataitem)
                  row+=1
                  worksheet.write_column(row, col, value)
                  col+=1
        except IOError as err:
            print('Data write error:'+str(err))
            return False
        else:
            workbook.close()
            print('Successfully finished writing')
            return True

    def xlsMultilineWriter(self,filedir,filename,sheetname,data,nsp):
        """
        writer the input data into a xls file with given filedir and file name
        :param filedir:The path for file where will be stored
        :param filename:The file name
        :param data:narray
        :return:bool type True when data have been successfully stored
        """
        workbook = xlwt.Workbook(encoding = 'ascii')
        worksheet1 = workbook.add_sheet(sheetname+'deviation')
        worksheet2 = workbook.add_sheet(sheetname+'mean')
        dindex=0
        worksheet1.write(dindex,0,'VoumeID.')
        worksheet1.write(dindex,1,'ImageID')
        worksheet1.write(dindex,2,'superpixel Deviation')

        worksheet2.write(dindex,0,'VoumeID.')
        worksheet2.write(dindex,1,'ImageID')
        worksheet2.write(dindex,2,'superpixel Mean')
        cols=1
        for i in range(len(nsp)):
            cols+=1
            worksheet1.write(1,cols,nsp[i])
            worksheet2.write(1,cols,nsp[i])
        dindex+=1
        for d in data.tolist():
            dindex+=1
            cols=0
            worksheet1.write(dindex,0,d['volumeID'])
            worksheet1.write(dindex,1,d['imageID'])

            worksheet2.write(dindex,0,d['volumeID'])
            worksheet2.write(dindex,1,d['imageID'])
            # print('The superpixel deviation:')
            # print(d['spdeviation'])
            # print('The superpixel mean:')
            # print(d['spmean'])
            cols+=2
            i=0
            for spdev,spmean in zip(d['spdeviation'],d['spmean']):

                  worksheet1.write(dindex,cols,spdev)
                  worksheet2.write(dindex,cols,spmean)
                  cols+=1
                  i+=1
        try:
           workbook.save(os.path.join(filedir,filename))
        except IOError as err:
            print('Data write error:'+str(err))
            return False
        else:
            print('Successfully finished writing')
            return True
#******************Batch data save*********************************************
    def BatchSave(self,filedir,fname,BatchArray):

       filepath=os.path.join(filedir,fname)
       np.save(filepath,BatchArray)
     # filename=os.path.join(filedir,"Mia")
     #pen(filename, "wb").write(contents)
#******************information data save*********************************************
    def InfoSaveTxt(self,filedir,info):
        """
        For test file write
        :param filedir:The path is to be stored
        :param info:The array data for writing
        :return:True when file have been successfully writen
        """
        try:
         f=open(filedir,'w')
         for i in range(0,len(info)):
            #print(info,file=f)
             f.write(info[i]+'\n')
        except IOError as err:
             print('File Error:'+str(err))
             return False
        finally:
            f.close()
            return True
      #return True
#Nrrd 3d format data write
    def VolSave(self, filedir, fname, VolData):
        """
        write data into 3D format file
        :param filedir: The file path where will be stored
        :param fname: The file name
        :param VolData:volume data
        :return:True when file have been successfully writen
        """
        filepath = os.path.join(filedir, fname)
        try:
            itk.WriteImage(itk.GetImageFromArray(VolData),filepath)
        except IOError as err:
             print('File Error:'+str(err))
             return False
        else:
            print('Successfully finished writing')
            return True
    def Imagesave(self,filedir,fname,imgdata):
        """

        :param filedir: The file path where will be stored
        :param fname: The file name
        :param imgdata: ndarray data
        :return:
        """
        filepath = os.path.join(filedir, fname)
       # img=self.to_rgb(imgdata)
        try:
            imsave(filepath,imgdata)
        except IOError as err:
             print('File Error:'+str(err))
             return False
        else:
            print('Successfully finished writing')
            return True



    def to_rgb(self,img):
      """
        Converts the given array into a RGB image. If the number of channels is not
        3 the array is tiled such that it has 3 channels. Finally, the values are
        rescaled to [0,255)

        :param img: the array to convert [nx, ny, channels]

        :returns img: the rgb image [nx, ny, 3]
      """
      img = np.atleast_3d(img)
      channels = img.shape[2]
      if channels < 3:
        img = np.tile(img, 3)

      img[np.isnan(img)] = 0
      # img -= np.amin(img)
      # img /= np.amax(img)
     # img *= 255
      return img
