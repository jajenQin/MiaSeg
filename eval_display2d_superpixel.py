# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
import platform
import logging
from matplotlib import pyplot as plt
import numpy as np

from skimage.measure import regionprops
from skimage.segmentation import slic
from datasets import dataset_factory
from datasets.DataInput import Miaconvert_medical
from datasets.DataInput import MiaDataReader

from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

Labelnames=[
            ]
tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
if platform.system() == 'Windows':
# #windows os
  tf.app.flags.DEFINE_string(
    'checkpoint_path','D:\DATA\Results\liver\sp3classesSur\modelNtumor\model2000sp0.5multi1',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
  tf.app.flags.DEFINE_string(
    'eval_dir', 'D:\DATA\Results\liver\sp3classesSur\modelNtumor\model2000sp0.5multi1', 'Directory where the results are saved to.')
  tf.app.flags.DEFINE_string(
    'dataset_dir', 'D:\DATA\Results\liver\sp3classesSur\datasetNtumor', 'The directory where the dataset files are stored.')

elif platform.system() == 'Linux':
#linux os
  tf.app.flags.DEFINE_string(
    'checkpoint_path','/home/svn/svncheckout/stanford/data/liver/tfmodel',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
  tf.app.flags.DEFINE_string(
    'eval_dir','/home/svn/svncheckout/stanford/data/liver/tfmodel', 'Directory where the results are saved to.')

  tf.app.flags.DEFINE_string(
     'dataset_dir', '/home/svn/svncheckout/stanford/data/liver', 'The directory where the dataset files are stored.')


tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'medical', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'file_name', 'liverSP2d0_%s.tfrecord', 'The file name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'eval', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'medicalnet', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'lenet', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 32, 'Eval image size')
tf.app.flags.DEFINE_integer(
    'eval_sample_number', 2000,'Eval image size')
Volshape=[]
FLAGS = tf.app.flags.FLAGS
# # pseudocolor for display
# def color_image(image, num_classes=9):
#     import matplotlib as mpl
#     norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
#     mycm = mpl.cm.get_cmap('Set1')
#     return mycm(norm(image))

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler=logging.FileHandler(os.path.join(FLAGS.dataset_dir,'Eval_Datareader.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
CLASS_NAMES = [
         'Others',
         'liver',
        'ambiguity'
             ]
Labeltype={
       'liver': 1
           }
# Labeltype={
#            'BrainStem':1,
#            'Chiasm':2,
#            'Mandible':3,
#            'OpticNerve_L':4,
#            'OpticNerve_R':5,
#            'Parotid_L':6,
#            'Parotid_R':7,
#            'Submandibular_L':8,
#            'Submandibular_R':9
#
#            }
background={'Others':0}


# for data normalized caculation
def DataNormalize(dataarray):
    # input narray
    # return normalize data with max and min
    Minpixel = min(dataarray.ravel())
    Maxpixel = max(dataarray.ravel())
    factor = 0.00001
    PatchNorm = np.divide(((dataarray.ravel() - Minpixel) * 255), ((Maxpixel - Minpixel) + factor))
    return PatchNorm.reshape(dataarray.shape)

def GenerateLabel(label):
    # input requires array
    size = len(label)
    label_all_image = label[0]
    for i in range(1, size):
        label_all_image = label_all_image + label[i]

    return label_all_image
def DataReader(filedir,fname):
    datareader = MiaDataReader.MiaDataReader(logger)
    labelreader = MiaDataReader.MiaDataReader(logger)

    # read the data files and random it to get train and test datasets
    datalist = datareader.GetFilelist('D:\stanford\MiaSeg1.0\MiaSeg\datasets\DataInput\imagetest.txt')  # get the data list
   # datalist = datareader.GetFilelist('/home/svn/svncheckout/stanford/MiaSeg/datasets/DataInput/imagetest.txt')  # get the data list

    datacount = 0
    # datasets=[]
    # labelsets=[]

    for datapath in datalist:
        print('******************************************************************')
        datacount += 1
        print('The %dth / %d Data volume start to load' % (datacount, len(datalist)))
        datareader.SetInputData(datapath)
        # datareader.SetInputData('.\..\..\data\HeadandNeck\c0001\img.nrrd')
        datareader.update()
##************label data read from files**********************
            #labelreader.SetInputData('.\datatest\label\Mandible.nrrd')
        labellist = []
        for labelname in Labeltype.keys():
                #labelpath='structures\\'+labelname+'.nii'
                labelpath='segmentation-'+os.path.split(datapath)[-1].split('-',1)[-1]
                labelpath=os.path.join(os.path.split(datapath)[0],labelpath)
                if os.path.exists(labelpath):
                    print("%s label data is being read" % labelname)
                    labelreader.SetInputData(labelpath)
                    labelreader.update()
                    labelvalue = Labeltype[labelname]
                    labeldata = labelreader.GetArray()
                   # labelindex = (labeldata==1).nonzero()  # Get the label organ region location
                   # labeldata[labelindex] = labelvalue  # set each organ with different label value
                    labellist.append(labeldata)
        print('*********************************************************************')

        label_all = GenerateLabel(np.array(labellist))
       # datareader.VolSave(filedir,fname,label_all)
       # datasets.append(datareader.GetArray())
       # labelsets.append(label_all)
    return datareader,label_all


def SuperpixelPatchWriter(img,labelarray,stride,numsuperpixels,datadir,tfwriter):
    """
    #Extract each supervoxel patch and write into one tfrecord format
    :param vols:ndarray of image data
    :param labelarray:ndarray of label data
    :param stride: the patch size
    :param datadir:The data path that is stored
    :param tfrecord_writer:The writer handler for tf format write

    :return: number of patch
    """
    stridecenter = int(stride / 2)
    count = 0
    #convert to range at 0-1
    factor = 0.00001
    PatchNorm = np.divide(((img.ravel() - min(img.ravel()))),
                              ((max(img.ravel()) - min(img.ravel())) + factor))

    PatchNorm= PatchNorm.reshape(img.shape)
    from skimage.filters import gaussian
    PatchNorm = gaussian(PatchNorm, sigma=0.5)

    slice_colors = np.dstack((np.dstack((PatchNorm, PatchNorm)), PatchNorm))
    superpixel = slic(slice_colors, n_segments=numsuperpixels, compactness=5, sigma=1)#superpixel caculation wiht slic
    #slice_entro = entropy(PatchNorm, disk(5))#get the whole image entropy
    region = regionprops(superpixel,img)#get the all regions information
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    ax[0].imshow(slice_colors)
    from skimage.segmentation import mark_boundaries
    ax[1].imshow(mark_boundaries(slice_colors, superpixel))
    ax[1].imshow(labelarray,alpha=0.5,cmap=plt.cm.gray)
    ax[2].imshow(labelarray)
    ax[3].imshow(superpixel)
    total=len(region)
    imgs_list = []
    regionindex=[]
    labelvalue=[]
    EvalCord=[]
    for r in range(len(region)):

        ymin=int(region[r].centroid[0])- stridecenter
        ymax=int(region[r].centroid[0]) + stridecenter
        xmin=int(region[r].centroid[1]) - stridecenter
        xmax=int(region[r].centroid[1]) + stridecenter
        if ymin>=0 and xmin >=0 and ymax<img.shape[0] and xmax<img.shape[1]:
            regionindex.append(region[r].label)#save the region index that is extracted
            flaglabel=(labelarray[ymin:ymax,xmin:xmax] == 1).sum()/labelarray[ymin:ymax,xmin:xmax].size
            if flaglabel==1:
                 label =1
            elif flaglabel>=0.5 and flaglabel<1:
                 label=3
            else:
                label=0
            EvalCord.append(np.array([int(region[r].centroid[0]),int(region[r].centroid[1])]))# for test patches extration
            imgs_list.append(np.hstack((label,DataNormalize(img[ymin:ymax,xmin:xmax]).ravel())))
            count += 1
            labelvalue.append(label)

    Miaconvert_medical.Convert2Tfrecord(datadir,np.array(imgs_list),tfrecord_writer=tfwriter,classname=CLASS_NAMES)
                # datareader.BatchSave(datadir,'%dVoxsel'%count,np.array(vols_list))

    print('The %d@%d superpixel region patch have done' % (count, total))
    imgs_list = np.array(imgs_list)[:,1:np.array(imgs_list).shape[1]]
    shape = [imgs_list.shape[0], stride, stride]
    vols = imgs_list.reshape(shape)
    #Miaimshow.subplots(vols[0:39], num=2, cols=6)

    if count != len(regionindex):
        raise ValueError('The superpixel extraction is wrong ')
    return regionindex,superpixel,labelvalue


def EvalSample_Gnerate(datadir,filename):
    #fname='Glabel.nrrd'
    datareader,labels=DataReader(datadir,filename)
    #vols=datareader.GetArray()

    borderindex = datareader.BorderImage(labels.nonzero())
    vols = datareader.GetArray()[borderindex[0]:borderindex[1],
             :,
             :
             ]
        #adjust the window-level to enhance organ region
    vols[vols>200]=200
    vols[vols<-200]=-200
    #borderindex[0]-16:borderindex[1]+16
    #vols = datareader.DataNormalize(vols)  # Normalize the data
    if os.path.exists(os.path.join(datadir,  'OrgVol.nrrd')):
        os.remove(os.path.join(datadir,  'OrgVol.nrrd'))

    datareader.VolSave(datadir, 'OrgVol.nrrd', vols)
    labelarray = labels[borderindex[0]:borderindex[1],
             :,
             :
             ]
    if os.path.exists(os.path.join(datadir, filename)):
        os.remove(os.path.join(datadir, filename))
    datareader.VolSave(datadir, filename, labelarray)
    eval_filename = os.path.join(datadir, FLAGS.file_name%FLAGS.dataset_split_name)
    # if the files existed
    if  tf.gfile.Exists(eval_filename):
        tf.gfile.Remove(eval_filename)
        print('Dataset files already exist. has removed it before re-creating them.')
        #return

    # *************************************************************************#
    #              convert data into TFrecord format
    # *************************************************************************#

    vols_list = []
    stride = 32
    # image patch extractied
    stridecenter = int(stride / 2)
    count=0
    total=(vols.shape[0] - stride)*(vols.shape[1] - stride)*(vols.shape[2] - stride)
    with tf.python_io.TFRecordWriter(eval_filename) as tfrecord_writer:
       #for i in range(vols.shape[0]):# extract each slice along the axis=0
         #superpixel-based patches extracted
         img,label,border=datareader.SalientsliceExtract(vols,labelarray)
         regionindex,supermap,labelval=SuperpixelPatchWriter(img,label,stride,FLAGS.eval_sample_number,datadir,tfrecord_writer)
         #Patchsize=VoxelPatchWriter(vols, labelarray, stride, datadir, tfrecord_writer)
        #RegionPatchWriter(vols, labelarray, stride, datadir, tfrecord_writer)

    return datareader,vols.shape,regionindex,supermap,label,img

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()
    ######################
    # Gnerate the tfRecorder data #
    ######################
    datareader,Volshape,rindex,spmap,labelOrg,imgOrg=EvalSample_Gnerate(FLAGS.dataset_dir, 'Glabel.nrrd')
    Patchsize=len(rindex)
    ######################
    # Select the dataset #
    ######################

    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir,file_pattern=FLAGS.file_name,Datasize=Patchsize)

    ####################
    # Select the model #
    ####################
    #num_classes=2

    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=(dataset.num_classes - FLAGS.labels_offset),
                is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    # label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                 preprocessing_name,
                 is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
             [image, label],
             batch_size=FLAGS.batch_size,
             num_threads=FLAGS.num_preprocessing_threads,
             capacity=5 * FLAGS.batch_size)

     ###################
    # Define the model #
    ####################

    logits, end_points = network_fn(images)
    probabilities = tf.nn.softmax(logits)
    pred = tf.argmax(logits, dimension=1)
    # if FLAGS.moving_average_decay:
    #   variable_averages = tf.train.ExponentialMovingAverage(
    #       FLAGS.moving_average_decay, tf_global_step)
    #   variables_to_restore = variable_averages.variables_to_restore(
    #       slim.get_model_variables())
    #   variables_to_restore[tf_global_step.op.name] = tf_global_step
    # else:
    #   variables_to_restore = slim.get_variables_to_restore()

    # #predictions = tf.argmax(logits, 1)
    # labels = tf.squeeze(labels)
    #
    # # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #     'Recall@5': slim.metrics.streaming_recall_at_k(
    #         logits, labels, 5),
    # })
    #
    # # Print the summaries to screen.
    # summary_ops = []
    # for name, value in names_to_values.items():
    #   summary_name = 'eval/%s' % name
    #   op = tf.scalar_summary(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    #   summary_ops.append(op)
    # # TODO(sguada) use num_epochs=1
    # if FLAGS.max_num_batches:
    #   num_batches = FLAGS.max_num_batches
    # else:
    #   # This ensures that we make a single pass over all of the data.
    #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
            checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)
    init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(checkpoint_path),
                slim.get_model_variables())# vriable name???

    #Volshape=(50,61,61)
    imgshape=spmap.shape
    segResult=np.zeros(imgshape)
    groundtruth=np.zeros(imgshape)
    segPromap=np.zeros(imgshape)
    segPromapEdge=np.zeros(imgshape)
    PreMap=[]
    labellist=[]
    seglist=[]
    conv1list=[]
    conv2list=[]
    imgorglist=[]
    fclist=[]
    with tf.Session() as sess:
            # Load weights
            init_fn(sess)
           # sess.run(images.initializer, feed_dict)
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            num_iter = int(math.ceil(dataset.num_samples/float(FLAGS.batch_size)))
            step = 0

            try:
                while step < num_iter and not coord.should_stop():
                    # Run evaluation steps or whatever
                    segmentation,log,pmap,labelmap,imgpre,conv1,conv2,fc3= sess.run([pred,logits,probabilities,labels,images,end_points['conv1'],end_points['conv3'],end_points['fc3']])
                    step+=1
                    PreMap.extend(pmap)
                    conv1list.extend(conv1)
                    conv2list.extend(conv2)
                    fclist.extend(fc3)
                    imgorglist.extend(imgpre)
                    seglist.append(segmentation)
                    labellist.append(labelmap)
                    print('The No. %d/%d caculation'%(step,num_iter))
                    #Miaimshow.subplots(Pro_imgs, num=step+2, cols=8)
            except tf.errors.OutOfRangeError:
                print('Done evalutaion -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            PreMap = np.array(PreMap)
            np.save(os.path.join(FLAGS.dataset_dir,'liverspProbmap.npy'), PreMap)
            # PreMap=np.squeeze(PreMap,axis=1)

           # PreMap_flat = PreMap.ravel()
           # PreMap_flat=np.divide((PreMap_flat - np.amin(PreMap_flat)) * 255, (np.amax(PreMap_flat) - np.amin(PreMap_flat)))
            m = 0
            for i in range(len(rindex)):
                segResult[spmap==rindex[i]]=np.array(seglist).ravel()[i]
                segPromap[spmap==rindex[i]]=PreMap[i,1]
                segPromapEdge[spmap==rindex[i]]=PreMap[i,2]
                groundtruth[spmap==rindex[i]]=np.array(labellist).ravel()[i]

            coord.join(threads)

            fig,ax= plt.subplots(nrows=2,ncols=3)

            from skimage.segmentation import mark_boundaries
            ax[0,0].set_title('Segmentation with superpixel map')
            ax[0,0].imshow(mark_boundaries(segResult, spmap))
            ax[0,1].set_title('Segmentation with ground truth map')
            ax[0,1].imshow(segResult)
            ax[0,1].imshow(labelOrg,alpha=0.5,cmap='jet')
            ax[0,2].set_title('Reading label')
            ax[0,2].imshow(groundtruth)

            ax[1,0].set_title('liver Probabilities map')
            ax[1,0].imshow(segPromap)
            ax[1,1].set_title('edge Probabilities map')
            ax[1,1].imshow(segPromapEdge)
            ax[1,2].set_title('liver +edge Probabilities map')
            ax[1,2].imshow(segPromapEdge+segPromap)

            segthpro=segPromapEdge+segPromap
            segthpro[segthpro<0.8]=0

    from skimage.segmentation import active_contour
    from skimage.measure import find_contours
    from skimage.filters import gaussian
    from skimage import morphology

    #edg=sobel(segResult.astype(int))
    segmorp=morphology.remove_small_objects(segthpro.astype(bool),5000)
    segopen=morphology.opening(segmorp,morphology.disk(3))
    segclose=morphology.closing(segopen,morphology.disk(15))
    fig,ax=plt.subplots(1,3)
    ax=ax.ravel()
    ax[0].imshow(segmorp)
    ax[0].set_title('Removed the small objects')
    ax[1].imshow(segopen)
    ax[1].set_title('After open operation')
    ax[2].imshow(segclose)
    ax[2].imshow(labelOrg,alpha=0.5,cmap='jet')
    ax[2].set_title('After close operation')
    plt.axis('off')
    from MiaUtils import Miametrics as metric
    mt=metric.MiaMetrics(logger)
    dsc=mt.DSCMetric(segclose,labelOrg.astype(bool))
    print('The dice similarity coefficient score is {}'.format(dsc))

    voe=mt.VOEMetric(segclose,labelOrg.astype(bool))
    print('The Volume overlap Error score is {}'.format(voe))

    rvd=mt.RVDMetric(segclose,labelOrg.astype(bool))
    print('The Relative voume difference score is {}'.format(rvd))

    from medpy.metric.binary import hd
    from medpy.metric.binary import asd
    from medpy.metric.binary import  obj_fpr
    from medpy.metric.binary import  obj_tpr
    Asd=asd(segclose,labelOrg.astype(bool))
    print('The Asd score is {}'.format(Asd))

    HD=hd(segclose,labelOrg.astype(bool))
    print('The Hausdorff Distance score is {}'.format(HD))
###************************************************************************
    ####superpixel-graph cuts mehtod computation
#######********************************************************************
    from skimage import segmentation, color,filters
    from skimage.future import graph
    img=DataNormalize(imgOrg)/255
    img=np.dstack((np.dstack((img, img)), img))
    labels1 = segmentation.slic(img, compactness=5, n_segments=2000,sigma=1)
    #labels1=spmap
    out1 = color.label2rgb(labels1, img, kind='avg')
    edge_map = filters.sobel(color.rgb2gray(img))
    g = graph.rag_boundary(labels1, edge_map)
    #g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg')

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    ax[0].imshow(out1)
    ax[1].imshow(out2)
    for a in ax:
       a.axis('off')
    plt.tight_layout()

if __name__ == '__main__':
  tf.app.run()

