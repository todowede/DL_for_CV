# import libraires
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential  # create simple feedforward nn
from keras.layers.core import Dense  # fully-connected layers
from tensorflow.keras.optimizers import SGD  # optimization
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
# the --output switch is the path to save plots of loss/accuracy
args = vars(ap.parse_args())

# grab the MNIST dataset
print("[INFO] loading MNIST(full) dataset...")
dataset = fetch_openml('mnist_784', as_frame=False)

# scale the raw pixek intensities to the range [0, 1.0], then
# construct the training and testing splits
data = dataset.data.astype("float") / 255.0  # data normalization
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  dataset.target,
                                                  test_size=0.25)

# Given the training and testing splits, encode labels
# convert the labels from integers to vectors (one-hot encoding)
# for train and test
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# lets define our network architecture
# we define a 784-256-128-10 arch using keras
model = Sequential()  # imply layer stacked on each other
# first fully-connected layer; weight = 256, input shape = 784 (28 x 28)
model.add(Dense(256, input_shape=(784, ), activation="sigmoid"))
# second layer; learns 128 weights
model.add(Dense(128, activation="sigmoid"))
# output later; learns 10 weights = no of classes
# softmax activation for normalized class probailities of each prediction
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
# initialize SGD optimizer with learning rate of 0.01
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
# call to .predict will return class label probabilities
# for every data point in testX.
# predictions variable will be a Numpy array with shape (X, 10)
predictions = model.predict(testX, batch_size=128)
# each entry in a given role is therefore a probaility. To determine
# the class with the largest probaility we call .argmax(axis=1)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
