# import libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
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

        # build LeNet-define first set of CONV => RELU => POOL layers
        # CONV layer = 20 filters (K) each of which are 5 x 5
        # (i.e square F x F filters)
        # same padding to ensure size output = input
        model.add(Conv2D(20, (5, 5), padding="same",
                  input_shape=inputShape))
        model.add(Activation("relu"))
        # apply a 2 x 2 pooling with 2 x 2 stride decreasing input
        # volume by 75%
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # the input volume can be flattened and a FC layer with 500
        # nodes applied
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # final softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return constryucted arch
        return model
