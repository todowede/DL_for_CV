# import libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from Image_Classification.nn import ShallowNet
from tensorflow.keras.optimizers import SGD  # optimization
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

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
opt = SGD(lr=0.01)
# instantiate the shallownet architecture with input images
# of 32 x 32 x 3; and for 3 class label.
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO} training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              # batch size of 32 means 32 images will be presented
              # to the model at a time.
              batch_size=32, epochs=40, verbose=1)

# evaluate model performance
print("[INFO} evaluating network...")
predictions = model.predict(testX, batch_size=32)
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
plt.show()
# plt.savefig(args["output"])
