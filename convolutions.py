# import libraries
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


# define the convolution method
def convolve(image, K):
    # get the spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for the out image, and "padding" the
    # borders of the input image so the spatial size (i.e.
    # width and height)are not reduced.
    # we apply padding to ensure the dimension of input image
    # is same as the output image
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # apply convolution to the image
    # loop over the inpur image, "sliding" the kenel across
    # each (x - y) coordinate from left-to-right and
    # top-to-botom.
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y) coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perfom the actual convolution by taking the element-wise
            # multiplication between ROI and the kernel, then summing
            # the matric
            k = (roi * K).sum()

            # store the convolved value in the output(x, y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

            # rescale the output image to be in the range [0, 255]
            output = rescale_intensity(output, in_range=(0, 255))
            output = (output * 255).astype("uint8")  # convert image back to
            # 8-bit int

            # return the output imsge
            return output


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--image", required=True,
                help="path to the input image")
# the output switch is the path to save plots of loss/accuracy
args = vars(ap.parse_args())

# Construct average blurring kennel used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# Construct a shapening filter
sharpen = np.array((
                   [0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]), dtype="int")

# construct a laplacian kernel to dectect edge-like regions
# of an image
laplacian = np.array((
                     [0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]), dtype="int")

# Sobel kernek to detect edge-like regions along
# both the x- and y axis
#
# constuct sobel x-axis kernel
sobelX = np.array((
                  [-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]), dtype="int")
#
# constuct sobel y-axis kernel
sobelY = np.array((
                  [-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]), dtype="int")

# construct the emboss kerenel
emboss = np.array((
                  [-2, -1, 0],
                  [-1, 1, 1],
                  [0, 1, 2]), dtype="int")

# Lump all kernel together to construct a kernel bank
# which is list of kernel to apply, using both the
# convolve function and opencv filter2D
kernelBank = (
              ("small_blur", smallBlur),
              ("large_blur", largeBlur),
              ("sharpen", sharpen),
              ("laplacian", laplacian),
              ("sobel_x", sobelX),
              ("sobel_y", sobelY),
              ("emboss", emboss)
)

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, K) in kernelBank:
    # apply the kernel to the grayscale image using both functiions
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    # show the original images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
