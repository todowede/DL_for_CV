# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from Image_Classification.nn import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# handles our data imbalance issue by computing the class weights:
# account for skew in the labeled data
#
# calculate the total number of images in each class and 
#initialize dictionary to store the class weights
classTotals = labels.sum(axis=0)
classWeight = dict()

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]
#classWeight = classTotals.max() / classTotals
# computes the total number of examples per class. In this case, 
# # classTotals will be an array: [9475, 3690] for “not smiling” and 
# “smiling”, respectively. We then scale these totals on Line 53 to 
# obtain the classWeight used to handle the class imbalance, 
# yielding the array: [1, 2.56]. This weighting implies that our 
# network will treat every instance of “smiling” as 2.56 instances of 
# “not smiling” and helps combat the class imbalance issue by amplifying 
# the per-instance loss by a larger weight when seeing “smiling” examples.

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                                  test_size=0.20, 
                                                  stratify=labels, 
                                                  random_state=42)

# Now to train LeNet
# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64,
              epochs=15, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
