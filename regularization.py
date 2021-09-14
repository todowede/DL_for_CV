# Import packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Image_Classification.preprocessing import SimplePreprocessor
from Image_Classification.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# Construct the argument parse and parse the arguments
# argparse is the “recommended command-line parsing module in the
# Python standard library.” It’s what you use to get command line
# arguments into your program.
#
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input datset")

args = vars(ap.parse_args())

# Get the list of images

print("[INFO} loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Intialize the image preprocessor, laod the dataset from disk,
# and reshape the data matric
sp = SimplePreprocessor(32, 32)
sd1 = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sd1.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training ans 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=5)

# we apply a few differnt types of regularization when trianing the
# SGDclassifier
for r in (None, "l1", "l2"):
    # train a SGD classifier using a softmax loss function and the
    # specified regularizaion function for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=10,
                          learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{} penalty accuracy: {:.2f}%".format(r, acc * 100))
