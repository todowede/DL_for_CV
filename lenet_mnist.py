# import libraries
from Image_Classification.nn import LeNet
from tensorflow.keras.optimizers import SGD  # optimization
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# grab the MNIST dataset
print("[INFO] loading MNIST(full) dataset...")
dataset = fetch_openml('mnist_784', as_frame=False)
data = dataset.data

# we going to reshape the data
# if we are using "channels first" ordering, then reshape
# the design matrix such that the matrix is
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
# ortherwise we are using channels last ordering, so
# the design matrix shape should be:
# num_samples x rows x columns x depth
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

# scale the raw pixek intensities to the range [0, 1.0], then
# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
                                                  dataset.target.astype("int"),
                                                  test_size=0.25,
                                                  random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# traiing the LeNet network
# initialize the eoptimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
# instantiate the shallownet architecture with input images
# of 28 x 28 x 1; and for 10 class label.
model = LeNet.build(width=28, height=28, depth=1, classes=10)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO} training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              # batch size of 128 means 128 images will be presented
              # to the model at a time.
              batch_size=128, epochs=20, verbose=1)

# evaluate model performance
print("[INFO} evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
