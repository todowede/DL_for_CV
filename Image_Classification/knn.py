# Import packages
from sklearn.neighhbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Image_Classification.preprocessing import preprocessor
from Image_Classification.datasets import datasetloader
from imutils import paths
import argparse