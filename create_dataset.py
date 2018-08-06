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

@InProceedings{Bambach_2015_ICCV,
author = {Bambach, Sven and Lee, Stefan and Crandall, David J. and Yu, Chen},
title = {Lending A Hand: Detecting Hands and Recognizing Activities in Complex Egocentric Interactions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}

"""


"""

Creates tfrecord dataset from a few different hand dataset folders that can easily be ingested by the object detection API.

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
import zipfile

from object_detection.utils import dataset_util

# downloads all raw data from datasets to be converted to tfrecords

def download_egohands_data():
    url = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
    wget.download(url, 'egohands_data.zip')
    F = zipfile.ZipFile('egohands_data.zip')
    F.extractall('raw_data/egohands')
    F.close()
    # os.rename('raw_data/egohands_data', 'raw_data/egohands')
    os.remove('egohands_data.zip')

output_folder = 'data'

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

def egohands_to_tfrecord():
    count = 0
    metafilepath = 'raw_data/egohands/metadata.mat'
    meta = sio.loadmat(metafilepath, struct_as_record=False, squeeze_me=True)
    writer = tf.python_io.TFRecordWriter(os.path.join(output_folder,
                                                      'egohands_0.tfrecords'))
    for vid in meta['video']:
        vidfolder = os.path.join("raw_data/egohands/_LABELLED_SAMPLES/", vid.video_id)
        for frame in vid.labelled_frames:
            count += 1
            if count % 1000 == 0:
                print("swapping writers...")
                writer.close()
                writer = tf.python_io.TFRecordWriter(
                    os.path.join(output_folder,
                                 'egohands_%d.tfrecords' % (count / 1000)))
                
            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            if len(frame.yourleft) > 0:
                bbox = to_bounding_box(frame.yourleft)
                ymins.append(bbox[0])
                xmins.append(bbox[1])
                ymaxs.append(bbox[2])
                xmaxs.append(bbox[3])

            if len(frame.yourright) > 0:
                bbox = to_bounding_box(frame.yourright)
                ymins.append(bbox[0])
                xmins.append(bbox[1])
                ymaxs.append(bbox[2])
                xmaxs.append(bbox[3])

            if len(frame.myleft) > 0:
                bbox = to_bounding_box(frame.myleft)
                ymins.append(bbox[0])
                xmins.append(bbox[1])
                ymaxs.append(bbox[2])
                xmaxs.append(bbox[3])

            if len(frame.myright) > 0:
                bbox = to_bounding_box(frame.myright)
                ymins.append(bbox[0])
                xmins.append(bbox[1])
                ymaxs.append(bbox[2])
                xmaxs.append(bbox[3])

            bbox = (xmins, ymins, xmaxs, ymaxs)
            write_to_record(writer, vidfolder, 'frame_%04d.jpg' % frame.frame_num, bbox)
    writer.close()

if __name__ == "__main__":    
    if not os.path.isdir('raw_data'):
        print("Downloading datasets... This could take a while.")
        os.makedirs('raw_data')
        download_egohands_data()
        print("Done downloading datasets")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("egohands dataset:")
    egohands_to_tfrecord()
    print("done writing egohands dataset records")
    print("create labels.pbtxt:")
    with open('data/labels.pbtxt', 'w+') as F:
        F.write('item {\n  name: "hand"\n  id: 1\n  display_name: "hand"\n}')
    print("done.")
