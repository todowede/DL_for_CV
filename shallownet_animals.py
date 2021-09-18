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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

# get list of images
print("[INFO] loading images... ")
imagePaths = list(paths.list_images(args["dataset"]))

# now we create pipeline to load and preprocess out dataset
#
# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load dataset from disk, then scale the raw pixel intensitiesol
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partitions the data into training and testing splits using 75% for
# training and 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)

# cnvert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO} compiling model...")
opt = SGD(lr=0.005)
# instantiate the shallownet architecture with input images
# of 32 x 32 x 3; and for 3 class label.
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO} training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              # batch size of 32 means 32 images will be presented
              # to the model at a time.
              batch_size=32, epochs=100, verbose=1)

# evaluate model performance
print("[INFO} evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=["cat", "dog", "panda"]))

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
plt.show()
# plt.savefig(args["output"])
