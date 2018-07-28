import tensorflow as tf
import glob
import os
import cv2

tf.enable_eager_execution()

"""

small helper function that reads the data from the training tfrecords
to ensure they are formatted correctly...

"""

input_folder = '../data'

def get_filenames(folder):
    filenames = []
    for filename in glob.glob(os.path.join(folder, '*.tfrecords')):
        filenames.append(filename)
    return filenames

def get_image(record):
    feature_dict = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.VarLenFeature(tf.string),
        'image/source_id': tf.VarLenFeature(tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.VarLenFeature(tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64)
    }

    features = tf.parse_single_example(record, features=feature_dict)
    img = tf.image.decode_jpeg(features['image/encoded']).numpy()
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin']).numpy()
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax']).numpy()
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin']).numpy()
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax']).numpy()

    width = tf.cast(features['image/width'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    
    for i in range(len(xmin)):
        x1 = int(round(xmin[i] * width.numpy()))
        y1 = int(round(ymin[i] * height.numpy()))
        x2 = int(round(xmax[i] * width.numpy()))
        y2 = int(round(ymax[i] * height.numpy()))
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print((x1, y1, x2, y2))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    cv2.waitKey(1000)


if __name__ == "__main__":
    F = get_filenames(input_folder)
    D = tf.data.TFRecordDataset(F)
    D = D.shuffle(10000)
    it = D.make_one_shot_iterator()
    while True:
        get_image(it.get_next())
