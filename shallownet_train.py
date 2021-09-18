# import libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Image_Classification.preprocessing import SimplePreprocessor
from Image_Classification.datasets import SimpleDatasetLoader
from Image_Classification.preprocessing import ImageToArrayPreprocessor
from Image_Classification.nn import ShallowNet
from tensorflow.keras.optimizers import SGD  # optimization
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)