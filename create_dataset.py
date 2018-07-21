
"""

Creates dataset from downloaded hand_dataset folders

"""

import scipy.io as sio
import json
import os
import glob
import cv2

annotation_folder = 'hand_dataset/training_dataset/training_data/annotations'
image_folder = 'hand_dataset/training_dataset/training_data/images'

def write_to_record(img_folder, img_filename, boxes):
    xmins, ymins, xmaxs, ymaxs = boxes

    im = cv2.imread(os.path.join(img_folder, img_filename))
    width, height, channels = im.shape
    
    xmins[:] = [i / width for i in xmins]
    ymins[:] = [i / height for i in ymins]
    xmaxs[:] = [i / width for i in xmaxs]
    ymaxs[:] = [i / height for i in ymaxs]
    
    print(width, height, channels)
    

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

    return (xmin, ymin, xmax, ymax)

for filename in glob.glob(os.path.join(annotation_folder, '*.mat')):
    imgname = filename[len(annotation_folder) + 1:-3] + "jpg"
    print(imgname)

    """
    cv2.imshow('image', img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    """

    F = sio.loadmat(filename)
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
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
    write_to_record(image_folder, imgname, bbox)
    del F
    print("")
