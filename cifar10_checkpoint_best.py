# import libraries
from sklearn.preprocessing import LabelBinarizer
from Image_Classification.nn import MiniVGGNet
from tensorflow.keras.optimizers import SGD  # optimization
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
import argparse
# Because of large memory requirement, we have
# to grow the GPU in memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to bet model weights file")
args = vars(ap.parse_args())

# load the training and test data, then scale it
# into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimizer and model
print("[INFO} compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validaion loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss",
                             save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
print("[INFO} training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              # batch size of 64 means 64 images will be presented
              # to the model at a time.
              batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

# Now we can run the script and get the best model saved.
# python cifar10_checkpoint_best.py --weights
#                    test_best/cifar10_best_weights.hdf5
