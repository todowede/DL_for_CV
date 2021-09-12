# Import packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance  (-1 =available cores)")
args = vars(ap.parse_args())

# Get the list of images

print("[INFO} loading images...")
imagePaths = list(paths.list_images(args["datset"]))

# Intialize the image preprocessor, laod the dataset from disk,
# and reshape the data matric
sp = SimplePreprocessor(32, 32)
sd1 = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sd1.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Show info on memory consumption of the images
print("[info] features matrix: {:.1f}MB".format(
      data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training ans 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
                            target_names=le.classes_))
