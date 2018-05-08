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
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils



_IMAGE_SIZE = 32
_NUM_CHANNELS = 1
_NUM_TRAIN_FILES=4
_NUM_TEST_FILES=1
# The names of the classes.
_CLASS_NAMES = [
    'Others',
    'BrainStem',
    'Chiasm',
    'Mandible',
    'OpticNerve_L',
    'OpticNerve_R',
    'Parotid_L',
    'Parotid_R',
    'Submandibular_L',
    'Submandibular_R']


def _extract_images(filename, num_images):
    """Extract the images into a numpy array.

    Args:
      filename: The path to an MNIST images file.
      num_images: The number of images in the file.

    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    return data


def _extract_labels(filename, num_labels):
    """Extract the labels into a vector of int64 label IDs.

    Args:
      filename: The path to an MNIST labels file.
      num_labels: The number of labels in the file.

    Returns:
      A numpy array of shape [number_of_labels]
    """
    print('Extracting labels from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def _add_to_tfrecord(data_filename, tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
      data_filename: The filename of the MNIST images.
      labels_filename: The filename of the MNIST labels.
      num_images: The number of images in the dataset.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # images = _extract_images(data_filename, num_images)
    # labels = _extract_labels(labels_filename, num_images)
    data = np.load(data_filename)  # load numpy array data
    images = data[:, 1:data.shape[1]]
    num_images = images.shape[0]
    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    # for 3D DATA input
   # shape = (_IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_SIZE，_NUM_CHANNELS)
    #images = images.reshape((num_images, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_SIZE，_NUM_CHANNELS))
    images = images.reshape((num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS))
    labels = data[:, 0]

    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                sys.stdout.flush()

                png_string = sess.run(encoded_png, feed_dict={image: images[j]})

                example = dataset_utils.image_to_tfexample(
                    png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, int(labels[j]))
                tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/medical_vox2dROI%s.tfrecord' % (dataset_dir, split_name)


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        # data_filename = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
        # labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
        for i in range(_NUM_TRAIN_FILES):
            data_filename = os.path.join(dataset_dir,
                                         'patch',
                                         'Batch2DROIvoxbin_train%d.npy' % (i + 1))  # 1-indexed.
            _add_to_tfrecord(data_filename, tfrecord_writer)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        # data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
        # labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
        for i in range(1,_NUM_TEST_FILES):
           data_filename = os.path.join(dataset_dir,
                                     'patch',
                                     'Batch2Dvoxbin_test%d.npy'%i)
           _add_to_tfrecord(data_filename, tfrecord_writer)


# Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    #_clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the medical dataset!')
