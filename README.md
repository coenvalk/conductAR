# conductAR

## Info

This project uses computer vision to track hand motion and machine learning to interpret the intended effect of the hand motion, in terms of tempo, dynamics, and attack.

## Installation

this program doesn't necessarily need to installed, but all dependencies necessary to run the python script need to be met.
This program is written in python 2.7.

### Dependencies

#### Python Dependencies
`requirements.txt` has all of the python dependencies. Install them with
```
pip install -r requirements.txt
```
#### Tensorflow Object Detection API
To track hand movement, this project makes use of [tensorflow models'](https://github.com/tensorflow/models) [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Follow [this link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install the object_detection api, and make sure that the research and slim are in the PYTHONPATH while running this program.

## Usage
`python create_dataset.py` creates tfrecord dataset used to train hand tracking model. Not necessary unless used to extend or improve the hand tracking model. NOTE: this script creates approximately 450 MB of new data.

`python model_main.py --logtostderr --model_dir=./model/ --pipeline_config_path=model/pipeline.config `  re-trains the hand detection model. NOTE: must have TFRecord files created by running `python create_dataset.py`

`python conductAR.py` takes the included pre-trained model to track hand motion.
