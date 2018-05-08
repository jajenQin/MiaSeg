# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
#from six.moves import urllib
import tensorflow as tf

from MiaSeg.datasets.DataInput import Miadataset_utils



#global IMAGE_SIZE

NUM_CHANNELS = 1
NUM_TRAIN_FILES=4
# The names of the classes.
_CLASS_NAMES = [
       'Others',
        'liver',
        'ambiguity'
   ]


def extract_images(data, num_images,PatchSize,channels):
    """Extract the images into a numpy array.

    Args:
      data: numpy array data with images and label.
      num_images: The number of images in the file.

    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """

    if not isinstance(data, (list, tuple,dict,np.ndarray)):
      raise ValueError('data need to be a list or tuple or dict')

    print('Extracting images')
    if isinstance(data,list):
        imagelist=[]
        for img in data:
           imagelist.append(img['image'])

        images=np.array(imagelist)
        images=images.reshape((num_images, PatchSize, PatchSize, channels))

    elif isinstance(data,np.ndarray):
       images = data[:, 1:data.shape[1]]
       images = images.reshape((num_images, PatchSize, PatchSize, channels))

    return images


def extract_labels(data, num_labels):
    """Extract the labels into a vector of int64 label IDs.

    Args:
      data: The path to an MNIST labels file.
      num_labels: The number of labels in the file.

    Returns:
      A numpy array of shape [number_of_labels]
    """
    if not isinstance(data, (list, tuple,dict,np.ndarray)):
      raise ValueError('data need to be a list or tuple or dict')
    print('Extracting labels')

    if isinstance(data,list):
        labels=[]
        for label in data:
           labels.append(label['label'])

        labels=np.array(labels)
        labels=labels.reshape((num_labels,1))

    elif isinstance(data,np.ndarray):
      labels = data[:, 0]

    return labels
def extract_labels_full(data, num_labels,PatchSize,channels):
    """Extract the labels into a vector of int64 label IDs.

    Args:
      data: The path to an MNIST labels file.
      num_labels: The number of labels in the file.

    Returns:
      A numpy array of shape [number_of_labels]
    """
    if not isinstance(data, (list, tuple,dict,np.ndarray)):
      raise ValueError('data need to be a list or tuple or dict')
    print('Extracting labels')

    if isinstance(data,list):
        labels=[]
        for label in data:
           labels.append(label['label'])

        labels=np.array(labels)
        labels=labels.reshape((num_labels, PatchSize, PatchSize,1))


    return labels

def add_to_tfrecord(dataarray, tfrecord_writer,PatchSize=32,channels=1):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
      dataarray: The numpy array with images and labels.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    if not isinstance(dataarray, (list, tuple,dict,np.ndarray)):
      raise ValueError('data need to be a list or tuple or dict')

    if isinstance(dataarray,list):
        num_images=len(dataarray)

    elif isinstance(dataarray,np.ndarray):
        num_images=dataarray.shape[0]
   # global IMAGE_SIZE
   # IMAGE_SIZE=32

    images = extract_images(dataarray, num_images, PatchSize, channels)
    labels = extract_labels(dataarray, num_images)
    #data = np.load(data_filename)  # load numpy array data
    #images = data[:, 1:data.shape[1]]
   # num_images = images.shape[0]
    shape = (PatchSize, PatchSize,channels)
    # for 3D DATA input
   # shape = (_IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_SIZE，_NUM_CHANNELS)
    #images = images.reshape((num_images, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_SIZE，_NUM_CHANNELS))



    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                sys.stdout.flush()

                png_string = sess.run(encoded_png, feed_dict={image: images[j]})

                example = Miadataset_utils.image_to_tfexample(
                    png_string, b'png', PatchSize, PatchSize, int(labels[j]))
                tfrecord_writer.write(example.SerializeToString())

def add_to_tfrecord_full(dataarray,tfrecord_writer,PatchSize=32,channels=1):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
      dataarray: The numpy array with images and labels.
      tfrecord_writer: The TFRecord writer to use for writing.
      PatchSize: image size
      channels: number of channels
    """
    if not isinstance(dataarray, (list, tuple,dict,np.ndarray)):
      raise ValueError('data need to be a list or tuple or dict')

    if isinstance(dataarray,list):
        num_images=len(dataarray)

    elif isinstance(dataarray,np.ndarray):
        num_images=dataarray.shape[0]
    #global IMAGE_SIZE
    #IMAGE_SIZE=size

    images = extract_images(dataarray, num_images,PatchSize,channels)
    labels = extract_labels_full(dataarray, num_images,PatchSize,channels)
    #data = np.load(data_filename)  # load numpy array data
    #images = data[:, 1:data.shape[1]]
   # num_images = images.shape[0]
    shape = (PatchSize, PatchSize, channels)
    # for 3D DATA input
   # shape = (_IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_SIZE，_NUM_CHANNELS)
    #images = images.reshape((num_images, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_SIZE，_NUM_CHANNELS))



    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_image = tf.image.encode_png(image)
        label = tf.placeholder(dtype=tf.uint8, shape=(PatchSize,PatchSize,1))
        encoded_label = tf.image.encode_png(label)
        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                sys.stdout.flush()

                image_string = sess.run(encoded_image, feed_dict={image: images[j]})
                label_string = sess.run(encoded_label, feed_dict={label: labels[j]})

                example = Miadataset_utils.image_to_tfexample_full(
                    image_string, b'png', PatchSize, PatchSize,label_string)
                tfrecord_writer.write(example.SerializeToString())

def get_output_filename(dataset_dir, filename,split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      filename: The name of the train/test datasets
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s\%s_%s.tfrecord' % (dataset_dir, filename,split_name)


def Convert2Tfrecord(dataset_dir,dataarray,tfrecord_writer,PatchSize=32,channels=1,classname=None):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

   # training_filename = get_output_filename(dataset_dir, 'train')
   # testing_filename = get_output_filename(dataset_dir, 'test')

    # if tf.gfile.Exists(filename) :
    #     print('Dataset files already exist. Exiting without re-creating them.')
    #     return

    # First, process the  data:

    add_to_tfrecord(dataarray, tfrecord_writer,PatchSize=PatchSize,channels=channels)

# Finally, write the labels file:
    if classname:
        CLASS_NAMES=classname
    else:
        CLASS_NAMES=_CLASS_NAMES
    labels_to_class_names = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))
    Miadataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    #_clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the medical ROI voxel dataset!')

def Convert2TfrecordFull(dataset_dir,dataarray,tfrecord_writer,classname=None,PatchSize=32,channels=1):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

   # training_filename = get_output_filename(dataset_dir, 'train')
   # testing_filename = get_output_filename(dataset_dir, 'test')

    # if tf.gfile.Exists(filename) :
    #     print('Dataset files already exist. Exiting without re-creating them.')
    #     return

    # First, process the  data:

    add_to_tfrecord_full(dataarray, tfrecord_writer,PatchSize=PatchSize,channels=channels)

# Finally, write the labels file:
    if classname:
        CLASS_NAMES=classname
    else:
        CLASS_NAMES=_CLASS_NAMES
    labels_to_class_names = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))
    Miadataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    #_clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the medical ROI voxel dataset!')
