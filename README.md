# conductAR

## Info

This project uses computer vision to track hand motion and machine learning to interpret the intended effect of the hand motion, in terms of tempo, dynamics, and attack.

## Installation

this program doesn't necessarily need to installed, but all dependencies necessary to run the python script need to be met.
this program is written in python3, so make sure that is installed. installation has been tested with python 3.6.5, but should work with other python3.x versions.

### Dependencies

#### Python Dependencies
`requirements.txt` has all of the python dependencies. Install them with
```
pip install -r requirements.txt
```
#### Tensorflow Object Detection API
To track hand movement, this project makes use of [tensorflow models'](https://github.com/tensorflow/models) [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Follow [this link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install the object_detection api, and make sure that the research and slim are in the PYTHONPATH while running this program.

## Usage
`python3 create_dataset.py` creates tfrecord dataset used to train hand tracking model. Not necessary unless used to extend or improve the hand tracking model. NOTE: this script creates approximately 8gb of new data, so make sure that a decent amount of space is left on your device.

`python3 conductAR.py` takes the included pretrained model to track hand motion.
