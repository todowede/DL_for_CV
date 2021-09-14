# Import packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    # taje the dot product between our features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    # return the prediction
    return preds

# COnstruct the argument parse and parse the arguments


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1,000 data points
# where each data point is a 2D featire vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature
# matric -- this is the bias trick
X = np.c_[X, np.ones((X.shape[0]))]

# partition data into training and test splits at 50%
(trainX, testX, trainY, testY) = train_test_split(X, y,
                                                  test_size=0.5,
                                                  random_state=42)

# initiale the weight matrix and list of losses
print("[INFO} training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product btw fetaurs X and the weight matrix W
    # then pass this values throughh pur sigmoid fuction, thereby
    # giving prediction on the dataset
    preds = sigmoid_activation(trainX.dot(W))

    # we determine the error, which is the difference between
    # our predictions and the true values
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # we compute the gradient - the GD is the dot product
    # between our features and the error of the predictions
    gradient = trainX.T.dot(error)

    # we need to nudge the weight matrix in the negative direction
    # of the gradeint. tgis is where the gradient descent takes place
    W += -args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
                                                    loss))

# with classifier trained, now we evaluate the model
print("[INFO] evaluating...")

# make prediction
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# plot for loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
