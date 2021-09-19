# import libraries
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MiniVGGNet:
    @staticmethod
    # define build function that accepts width,height,depth
    # of image and no of clases to predict
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # channels last and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        # we introduce a "chanDim" variable - index of the channel
        # dimension. Batch Normalization operates over the channels
        # so to use BN, we need to know whic axis to normalize over.
        # chanDim = -1 measn the index of the channel dimension last
        # in the input shape (i.e channels last ordering)
        chanDim = -1

        # if we are using channels first update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # We define the first layer of MiniVGGNet, which is
        # first CONV => RELU => CONV => RELU => POOL layer set
        # (CONV => RELU => BN) * 2 => POOL => DO
        # same padding to ensure size output = input
        model.add(Conv2D(32, (3, 3), padding="same",
                  input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # default stride=pool size
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))  # here increase filter
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # default stride=pool size
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # final softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return constryucted arch
        return model
