# import package
import cv2


# Create Preprocessor class for the preprocessing functions.
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # define the constructor to the Preprocessor class
        # it requires two argument and optional third
        # width: target width of input image after resizing
        # height: target height if input image after resizing
        # inter: optional - interpolation algorithm to use
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # The preprocess function; take a single argument - input image
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)
