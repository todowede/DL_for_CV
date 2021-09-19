# set the matplotlib backend so that figures can be saved in the background
# this creates a non-interactive plot that will simple save to disk.
import matplotlib
matplotlib.use("Agg")

# import libraries
from Image_Classification.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from Image_Classification.nn import MiniVGGNet
from tensorflow.keras.optimizers import SGD  # optimization
from keras.datasets import cifar10
import argparse
import os
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
args = vars(ap.parse_args())

# show information on the process ID
#
# we use the processId assigned by the OS to name the
# plots and JSOn files. if the training is going poorly
# we can open the task manager and kill the process ID
# asociated wuth the scripts.
#
print("[INFO process ID: {}".format(os.getpid()))

# load the training and test data, then scale it
# into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
# he CIFAR-10 images are preprocessed and the channel
# ordering is handled automatically inside of cifar10.load_data,
# we do not need to apply any of our custom preprocessing classes.
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialze the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO} compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# we construct TrainingMonitor callback and train the network
# constructr the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(
                                                            os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(
                                                            os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO} training network...")
model.fit(trainX, trainY, validation_data=(testX, testY),
          batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
