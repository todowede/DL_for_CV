# set the matplotlib backend so that figures can be saved in the background
# this creates a non-interactive plot that will simple save to disk.
import matplotlib
matplotlib.use("Agg")

# import libraries
from sklearn.preprocessing import LabelBinarizer
from Image_Classification.nn import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD  # optimization
from sklearn.metrics import classification_report
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# Define custom learning rate
def step_decay(epoch):
    # initialize the base initial learning rate
    # drop factor and epochs to drop every
    initAlpha = 0.01
    factor = 0.25  # change to 0.5 to see effect
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)


# we now continue with the script
#
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

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

# we define the swt of callbacks to be passed to the model during
# training
callbacks = [LearningRateScheduler(step_decay)]

# initialize the optimizer and model
print("[INFO} compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
# the lr parameter will be ignored since we using LearningRateScheduler
# callback, so technically should be left out
#
# instantiate the shallownet architecture with input images
# of 32 x 32 x 3; and for 3 class label.
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO} training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              # batch size of 64 means 64 images will be presented
              # to the model at a time.
              batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

# evaluate model performance
print("[INFO} evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# plt.show()
plt.savefig(args["output"])
