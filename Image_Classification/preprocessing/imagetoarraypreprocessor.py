# import library
from keras.preprocessing.image import img_to_array


# define the constructor to our ImageToArrayPreprocessor class
class ImageToArrayPreprocessir:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the keras utility function that correctly rearranges
        # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)