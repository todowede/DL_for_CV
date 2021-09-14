# Import package
import numpy as np
import cv2

# Initialize class labels and set seed
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# randomly initialize the weight matrix and bias vector
# In real experiment - these parameters would be learned form the model

# our weight matrix has 3 rows (one for each clas labels)
# and 3072 columns (one for each pixels in our 32 x 32 x 3)
W = np.random.randn(3, 3072)
# bias vector has 3 rows (number of class labels) along one column
b = np.random.randn(3)

# Load image, resize it and flatten it into a "feature vector"
# representation
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

# compute the output scores by taking the dot product btw the weight matrix
# and the input image intensities, followed by adding the bias, b
scores = W.dot(image) + b  # scoring function

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)
