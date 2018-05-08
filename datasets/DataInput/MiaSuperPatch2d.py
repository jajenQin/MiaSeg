 #!/usr/bin/env python
import os
import sys
import vtk
from numpy import *
from PyQt5 import QtCore, QtGui,QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from datasets.DataInput import MiaDataReader as MiaDataReader
from datasets.DataInput import MiaDataPreprocess
import tensorflow as tf
import random as rand
from datasets.DataInput.Miaconvert_medical import *
import platform
import logging
from MiaUtils import MiaDataWriter
# tf.app.flags.DEFINE_string(
#      'dataset_dir', 'D:\DATA\Results\liver\sp3classesSur\PatchratioSe\data0.3', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
     'dataset_dir', 'D:\DATA\Results\liver\paperdemo\\test', 'The directory where the dataset files are stored.')

FLAGS = tf.app.flags.FLAGS
BATCHSIZE=10

Labeltype={
           'liver':1,
           }
background={'Others':0}


classname = [
     'Others',
     'liver',
    'ambiguity'
     ]

#'ambiguity'

class MiaSuperpixelPatch(QtWidgets.QMainWindow):
 
    def __init__(self, parent = None):

        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler=logging.FileHandler(os.path.join(FLAGS.dataset_dir,'Train_TestDatareader.log'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.testRate=0.2
        self.trainRate=1
        self.run()

    def run(self):
        datareader = MiaDataReader.MiaDataReader(self.logger)

        if platform.system()=='Windows':
            filedir =FLAGS.dataset_dir  # ????need to be set according to user input
            datafile='liverdata.xls'

        elif platform.system()=='Linux':
            filedir = './../../../data/liver'
            datafile = 'linuxliverdata.txt'
            training_filename = 'liverVox3d'
            testing_filename =  'liverVox3d'

        datalist=datareader.xlsReader(datafile,'test')
        numTrain=round(len(datalist)*self.trainRate)
        numTest=round(len(datalist)*self.testRate)
        flagtrain=False

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10)
        datalist=np.array(datalist)
        dindex=0
        filewriter=MiaDataWriter.MiaDataWriter()

        for trainindex, testindex in kf.split(datalist):
            dtrain=datalist[trainindex]
            print('The group %d/%d train data files:'%(dindex,kf.n_splits))
            print(dtrain)
            filewriter.xlsWriter(FLAGS.dataset_dir,'Trainfilegroup%d.xls'%dindex,'Trainfile',dtrain)
            dtest=datalist[testindex]

            print('The group %d/%d train data files:'%(dindex,kf.n_splits))
            print(dtest)
            filewriter.xlsWriter(FLAGS.dataset_dir,'Testfilegroup%d.xls'%dindex,'Testfile',dtest)

            #np.save(os.path.join(filedir,'trainfile%d.npy'%dindex),dtrain)
            #np.save(os.path.join(filedir,'testfile%d.npy'%dindex),dtest)
            training_filename = get_output_filename(filedir, 'liverSP2d%d'%dindex, 'train')
            testing_filename = get_output_filename(filedir, 'liverSP2d%d'%dindex, 'test')
      #First read training data and convert to Tfrecords
            filename='liverSP_train%d'%dindex
            self.DataRWExcute(filedir,training_filename,dtrain,filename)
     # Next read test data and convert to Tfrecords
            filename = 'liverSP_test%d'%dindex
            self.DataRWExcute(filedir, testing_filename, dtest,filename)
            dindex+=1

#Excute to Get ROI voxel patch wand save as numpy array as well as Tfrecords format
    def DataRWExcute(self,filedir,Tffilename,datalist,filename):
 ##************Raw data read from files**********************
        datareader=MiaDataReader.MiaDataReader(self.logger)
        labelreader=MiaDataReader.MiaDataReader(self.logger)
        patch=MiaDataReader.MiaDataReader(self.logger)
        BatchPatch=MiaDataPreprocess.MiaDataPreprocess()
        FlagVoxBatch=True

        BatchVoxbin=[]


        #datalength=len(datalist)
        datacount=0
        labelcount=0
        labelbincount=0
        backgtotalcount=0
        labeltotalcount = {
              'liver': 0,
              'Others': 0,
              'ambiguity':0
                   }
        stride = 32
        numSample = 300


        with tf.python_io.TFRecordWriter(Tffilename) as tfrecord_writer:
         for datapath in datalist:
            print('******************************************************************')
            self.logger.info('Starting to read %s'%datapath)
            print(datapath)
            datacount+=1
            # print('The %dth / %d Data volume start to load'%(datacount,len(datalist)))
            datareader.SetInputData(datapath)  
        #datareader.SetInputData('.\..\..\data\HeadandNeck\c0001\img.nrrd')
            datareader.update()

            DDims=datareader.GetOutputData().GetOutput().GetExtent()
            print("data size=[%d,%d,%d]"%(DDims[1],DDims[3],DDims[5]))

            Dspacing=datareader.GetOutputData().GetOutput().GetSpacing()
            print('data spacing=[%f,%f,%f]'%(Dspacing[0],Dspacing[1],Dspacing[2]))

            Dorigin=datareader.GetOutputData().GetOutput().GetOrigin()
            print('data origin=[%f,%f,%f]'%(Dorigin[0],Dorigin[1],Dorigin[2]))
        #datareader.SetImageSlice()

 ##************label data read from files**********************  
            #labelreader.SetInputData('.\datatest\label\Mandible.nrrd')
            labellist = []

            organcount=0
            for labelname in Labeltype.keys():
              labelpath='segmentation-'+os.path.split(datapath)[-1].split('-',1)[-1]
              labelpath=os.path.join(os.path.split(datapath)[0],labelpath)

              if os.path.exists(labelpath):
                self.logger.info("%s label data is being read" % labelname)
                print("%s label data is being read" % labelname)
                labelreader.SetInputData(labelpath)

                labelreader.update()
                labelvalue = Labeltype[labelname]
                labeldata = labelreader.GetArray()
                # labelindex = (labeldata==1).nonzero()  # Get the label organ region location
               # labeldata[labelindex] = labelvalue
                organcount += 1
                print("After convert into numpy Array")
                LDims = labelreader.GetOutputData().GetOutput().GetExtent()
                print("%s label size=[%d,%d,%d]" % (labelname, LDims[1], LDims[3], LDims[5]))

                Lspacing = labelreader.GetOutputData().GetOutput().GetSpacing()
                print('%s label spacing=[%f,%f,%f]' % (labelname, Lspacing[0], Lspacing[1], Lspacing[2]))

                Lorigin = labelreader.GetOutputData().GetOutput().GetOrigin()
                print('%s label origin=[%f,%f,%f]' % (labelname, Lorigin[0], Lorigin[1], Lorigin[2]))

                ##********************patch extraction according to label*****************************


                Superpatch,edgepatchsize=patch.Extract3DSuperPatch(datareader.GetArray(),labeldata,labelname,labeltotalcount,
                                                     numsuperpixels=numSample,stride=stride,flagslice=True,oneslice=True)

                patchsize=len(Superpatch)
                if patchsize != 0:

                    BatchVoxbin.extend(Superpatch)

                    print('*********************3D PATCH BIN PRINT*******************************')

                    print('######################################################################')
                    self.logger.info('Finisned extracting @%dth/%d label organ'%(organcount,len(Labeltype)))
                    self.logger.info('Finished %d super-pixel patch @ Volume %d extracted'%(patchsize,datacount))
                    print('Finisned extracting @%dth/%d label organ'%(organcount,len(Labeltype)))
                    print('Finished %d super-pixel patch @ Volume %d extracted'%(patchsize,datacount))
                    print('######################################################################')

                    labelcount += 1
                    labelbincount += 1

            print('*********************************************************************')
            print('                             segment line                            ')
            print('*********************************************************************')
            self.logger.info('The %dth/%d  volume data have been loaded'%(datacount,len(datalist)))
            print('The %dth/%d  volume data have been loaded'%(datacount,len(datalist)))
            print('*********************************************************************')


#save a batch for evry 10 volume and empty the Batchbin Array
            if datacount%BATCHSIZE==0:
               # fname='Batch2Dvoxbin'+str(int(datacount/BATCHSIZE))+'.npy'
                fname = filename+ '2d'+str(int(datacount / BATCHSIZE)) + '.npy'
                if BatchVoxbin:
                  # datareader.BatchSave(filedir,fname,array(BatchVoxbin))
                  # fname3d=filename+'3d'+str(int(datacount/BATCHSIZE))+'.npy'
                  # datareader.BatchSave(filedir,fname3d,array(BatchVox3Dbin))
                   Convert2Tfrecord(filedir,BatchVoxbin,tfrecord_writer,PatchSize=32,channels=1,classname=classname)
                else:
                   raise ValueError('There is no patch to be Extracted correctly')



                datainfor={'Numbers of Volume data':str(BATCHSIZE),
                           'Numbers of label organ':str(labelbincount),
                           'Numbers of 2D patch':str(len(BatchVoxbin)),
                          # 'Numbers of 3D patch':str(array(BatchVox3Dbin).shape),
                           'Numbers of chanel':str(datareader.Getchannel()),
                          # 'Patch Volume size':str(patch.Get3DRegionpatchArray().shape)
                           }
               # labeltotalcount.update({'Others': backgtotalcount})#add background count
                datainfor.update(labeltotalcount)#add labeltotal

                datareader.InfoSaveTxt(os.path.join(filedir,filename+str(int(datacount/BATCHSIZE))+'infor.txt'),self.DataInfo(datainfor))
                #BatchbinArray=array([])
                #Batchbin3DArray=array([])
                BatchVoxbin=[]
                #BatchVox3Dbin=[]
                datainfor={}
                labelbincount=0
                
            elif datacount==len(datalist):
                 #filedir='.\..\..\..\data\patch'
                # fname='Batch2Dvoxbin'+'test.npy'
                 fname = filename +'2d'+ str(datacount) + '.npy'
                 if BatchVoxbin:

                     Convert2Tfrecord(filedir, BatchVoxbin, tfrecord_writer,PatchSize=32,channels=1,classname=classname)
                 else:
                     raise ValueError('There is no patch to be Extracted correctly')

                # datareader.BatchSave(filedir,fname,Batchbin3DArray)
                 datainfor = {'Numbers of Volume data': str(BATCHSIZE),
                              'Numbers of label organ': str(labelbincount),
                              'Numbers of 2D patch': str(len(BatchVoxbin)),
                              'Numbers of chanel': str(datareader.Getchannel()),
                              }

                 #labeltotalcount.update({'Others': backgtotalcount})
                 datainfor.update(labeltotalcount)

                 datareader.InfoSaveTxt(os.path.join(filedir,filename+str(datacount)+'infor.txt'),self.DataInfo(datainfor))
                 BatchVoxbin = []
                 #BatchVox3Dbin = []
                 datainfor = {}

        print('%d label organs have been loaded'%labelcount)

#convert dict into string list
    def DataInfo(self,datainfo):
        #input requires a dict
        dataSave=list(datainfo.items())
        for i in range(len(dataSave)):
            dataSave[i]=str(dataSave[i][0])+':'+str(dataSave[i][1])
        return list(dataSave)

    #Produced seperated labels data into one label
    def GenerateLabel(self,label):
        #input requires array
        size=len(label)
        label_all_image=label[0]
        for i in range(1,size):
            label_all_image=label_all_image+label[i]

        return label_all_image
 
 
if __name__ == "__main__":
 
    app = QtWidgets.QApplication(sys.argv)
    widgets=QtWidgets.QWidget()
    window = MiaSuperpixelPatch()

    sys.exit(app.exec_())

