# The script used to create and train the model.

# check: https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
#  


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.utils import np_utils

import cv2
import numpy as np
import json

import tensorflow as tf
tf.python.control_flow_ops = tf

data_dir = './data/udacity_data/'
rows = 80
cols = 160
channels = 3
image_shape = (rows,cols)
batchSize = 32

# TODO: import the test data

def read_image(filename):
    newimage = cv2.imread(filename)
    newimage = cv2.resize(newimage,image_shape)
    newimage = cv2.cvtColor(newimage,cv2.COLOR_BGR2RGB)
    newimage = cv2.normalize(newimage,newimage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return newimage

def process_line(path,line):
    line = line.split(',')
    #center,left,right,steering,throttle,brake,speed

    #print('image filename: ', line[0], ' steering angle: ',line[3])
    image = read_image(path + '/' + line[0])
    #print(' image shape: ', image.shape)
    return image, line[3]


def generate_arrays_from_file(path):
    X = np.zeros((batchSize, cols, rows,channels))
    Y = np.zeros((batchSize, 1))
    batchIndex = 0
    while 1:
        l = 0
        #print('csv filename: ', path)
        f = open(path + '/driving_log.csv')
        for line in f:
            #print('X shape',X[batchIndex].shape)
            X[batchIndex], Y[batchIndex] = process_line(path,line)
            m = (batchIndex+1)%batchSize 
            #print(' line: {}, modulus: {}'.format(l,m))
            if (m== 0):
                batchIndex = 0
                yield X,Y
            batchIndex += 1
            l += 1
        f.close()



def test_generator():
    """Dummy routine to test the generator"""
    a
    c   = 0
    for X,y in generate_arrays_from_file('./data'):
        c += 1
    print(c)

    exit()

#test_generator()

# TODO: create the model

def create_nvidia_model():
    """The nividia model 
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()
    # Use a lambda layer to normalize the input data
    model.add(Lambda(
        lambda x: x/127.5 - 1.,
        input_shape=(cols, rows, channels),
        output_shape=(cols, rows, channels))
    )

    # Several convolutional layers, each followed by ELU activation
    # ELU http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf
    act = ELU

    # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(act())
    # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(act())
    # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    # Flatten the input to the next layer
    model.add(Flatten())
    # Apply dropout to reduce overfitting
    model.add(Dropout(.2))
    model.add(act())
    # Fully connected layer
    model.add(Dense(512))
    # More dropout
    model.add(Dropout(.5))
    model.add(act())
    # Fully connected layer with one output dimension (representing the speed).
    model.add(Dense(1))

    return model

def create_dummy_model():
    """Create a dummy model to rapidly test the surounding code
    """
    model = Sequential()
    model.add(Flatten(input_shape=(image_shape[1], image_shape[0], 3)))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

mymodel = create_dummy_model()



# TODO: train the model

mymodel.compile('adam', 'mean_squared_error', ['accuracy'])

# samples_per_epoch and batch size must be aligned
history = mymodel.fit_generator(generate_arrays_from_file(data_dir),  samples_per_epoch=256*100, nb_epoch=10, verbose=1)


print(history.history)
# TODO: check the accuracy

# export the model as required in the assignment
json_string = mymodel.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

# export the weights as required in the assignment
mymodel.save_weights('model.h5')


# Explicitly reap session to avoid an AttributeError sometimes thrown by
# TensorFlow on shutdown. See:
# https://github.com/tensorflow/tensorflow/issues/3388
from keras import backend as K
K.clear_session()

