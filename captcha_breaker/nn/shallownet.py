# import libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    # define build function that accepts width,height,depth
    # of image and no of clases to predict
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # channels last
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using channels first update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # build shallownet
        # define first (and only) CONV => RELU layer
        # CONV layer = 32 filters (K) each of which are 3 x 3
        # (i.e square F x F filters)
        # same padding to ensure size output = input
        model.add(Conv2D(32, (3, 3), padding="same",
                  input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return constryucted arch
        return model
