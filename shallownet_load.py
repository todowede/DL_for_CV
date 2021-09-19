# import libraries
from Image_Classification.preprocessing import SimplePreprocessor
from Image_Classification.datasets import SimpleDatasetLoader
from Image_Classification.preprocessing import ImageToArrayPreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
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
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# intialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images and randomly sample
# 10 image paths from the animal dataset for classification
#
# randomly sample indexes into the image paths list
print("[INFO] loading images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))  # get just 10 images
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disj then scale the raw pixel intensities
# to the range [0, 1]
sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sd1.load(imagePaths)
data = data.astype("float") / 255.0
# we have to preprocess the test images the exact same way we did
# during the training process.

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# now we can make prediciton on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
# Note thate .predict method of model will return a probabilities
# for every image in data - one probability for each class label. Taking
# the argmax on axis=1 finds the index of the class label with the
# larges propability for each image

# Visualize results
# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction and display it
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
