# Import libraries
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils  # imagenet submodule
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
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
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
                help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

# ensure a valid model name was supplied wia command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
                         "be a key in the `MODELS` dictionary")

# initialize the inoput image shape (224x224 pixels) aloing with the
# pre-processing function
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if using InceptionV3 or Xception, we need to set the
# input shape to (229x299) and use a different processing function
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

# load the network weights from disk (this will take time for first time for
# it to be downloaded, but subsequent run will be faster)
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# load input image using keras helper utiity while ensuring the image
# is resized to "inputShape", the required input dimensions for the
# ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through thenetwork. This is so bcos we train/calssify
# images in batches with CNN.
image = np.expand_dims(image, axis=0)

# preprocess the image using the appropriate function based on the model
# that has been loaded (i.e. mean subtraction, scaling, etc)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with ’{}’...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
# A call to .predict on Line 80 returns the predictions from the CNN. Given
# these predictions, # we pass them into the ImageNet utility function,
# decode_predictions, to give us a list of # ImageNet class label IDs,
# “human-readable” labels, and the probability associated with each class
# label.

# loop over the predictions and display the rank-5 predictions +
# probailities to our terminal
# The top-5 predictions (i.e., the labels with the largest probabilities) are
# then # printed to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image, and
# display the image to our screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)

# Classification Results
# Run the script and provide (1) path to input image to classify and (2) name
# of arch # to use. , e.g.
# python imagenet_pretrained.py --image \
# example_images/example_05.jpg --model resnet
