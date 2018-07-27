"""
MIT License

Copyright (c) 2018 Coen Valk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Acknowledgements:

[1] A. Mittal, A. Zisserman, P. H. S. Torr
Hand detection using multiple proposals  
British Machine Vision Conference, 2011 

"""


"""

Creates tfrecord dataset from hand_dataset data folders that can easily be ingested by the object detection API.

"""

import tensorflow as tf
import scipy.io as sio
import json
import os
import glob
import cv2
import numpy as np
import wget
import tarfile

from object_detection.utils import dataset_util

# downloads all raw data from datasets to be converted to tfrecords
def download_data():
    url = "http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz"
    wget.download(url, 'hand_dataset.tar.gz')
    F = tarfile.open('hand_dataset.tar.gz')
    F.extractall('raw_data')
    os.rename("raw_data/hand_dataset", "raw_data/oxford")
    os.remove('hand_dataset.tar.gz')

if not os.path.isdir('raw_data'):
    print("Downloading datasets... This could take a while.")
    os.makedirs('raw_data')
    download_data()

annotation_folders = ['raw_data/oxford/training_dataset/training_data/annotations',
                      'raw_data/oxford/validation_dataset/validation_data/annotations',
                      'raw_data/oxford/test_dataset/test_data/annotations']
image_folders = ['raw_data/oxford/training_dataset/training_data/images',
                 'raw_data/oxford/validation_dataset/validation_data/images',
                 'raw_data/oxford/test_dataset/test_data/images']
types = ['train', 'train', 'test']
output_folder = 'data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def normalize(values):
    new_values = []
    for v in values:
        if v < 0:
            v = 0
        elif v > 1:
            v = 1

        new_values.append(v)
    return new_values

def write_to_record(writer, img_folder, img_filename, boxes):
    xmins, ymins, xmaxs, ymaxs = boxes

    im = cv2.imread(os.path.join(img_folder, img_filename))

    """
    cv2.imshow("original", im)
    cv2.waitKey(500)
    """
    
    im = im.astype(np.float32)
    
    im_encoded = cv2.imencode('.jpg', im)[1]
    """
    with open('image.jpg', 'wb+') as f:
        f.write(im_encoded.tostring())
    """
 
    height, width, channels = im.shape

    xmins[:] = [i / width for i in xmins]
    ymins[:] = [i / height for i in ymins]
    xmaxs[:] = [i / width for i in xmaxs]
    ymaxs[:] = [i / height for i in ymaxs]

    xmins = normalize(xmins)
    ymins = normalize(ymins)
    xmaxs = normalize(xmaxs)
    ymaxs = normalize(ymaxs)

    """
    for i in range(len(xmins)):
        x1 = int(xmins[i] * width)
        y1 = int(ymins[i] * height)
        x2 = int(xmaxs[i] * width)
        y2 = int(ymaxs[i] * height)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    cv2.imshow("image", im)
    cv2.waitKey(500)
    """

    classes_text = ["hand".encode("utf8") for x in xmins]
    classes = [1 for x in xmins]


    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(im_encoded.tostring()),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    writer.write(example.SerializeToString())
    

def to_bounding_box(box):
    xmin = box[0][0]
    ymin = box[0][1]
    xmax = box[0][0]
    ymax = box[0][1]

    for point in box:
        if xmin > point[0]:
            xmin = point[0]
        if xmax < point[0]:
            xmax = point[0]
        if ymin > point[1]:
            ymin = point[1]
        if ymax < point[1]:
            ymax = point[1]

    return (ymin, xmin, ymax, xmax)
count = 0
for i in range(len(annotation_folders)):
    prefix = types[i]
    if prefix == 'test':
        count = 0
    print(prefix)
    annotation_folder = annotation_folders[i]
    image_folder = image_folders[i]
    writer = tf.python_io.TFRecordWriter(os.path.join(output_folder, prefix + '_0.tfrecords'))
    for filename in glob.glob(os.path.join(annotation_folder, '*.mat')):
        imgname = filename[len(annotation_folder) + 1:-3] + "jpg"
        
        F = sio.loadmat(filename)
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        count += 1
        if count % 1000 == 0:
            # swap writers.
            writer.close()
            writer = tf.python_io.TFRecordWriter(os.path.join(
                output_folder, prefix + '_' + str(count // 1000) + ".tfrecords"))
            print("swapping writers...")
        
        
        for box in F['boxes'].flatten():
            bbox = []
            for point in box.flatten()[0]:
                if point.shape == (1, 2):
                    bbox.append(point[0])
            rect = to_bounding_box(bbox)
            xmins.append(rect[0])
            ymins.append(rect[1])
            xmaxs.append(rect[2])
            ymaxs.append(rect[3])
        bbox = (xmins, ymins, xmaxs, ymaxs)
        write_to_record(writer, image_folder, imgname, bbox)
        del F
    writer.close()
