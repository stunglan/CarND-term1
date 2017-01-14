# The script used to create and train the model.

# check: https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
#  


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU

import cv2
import numpy as np
import json

import tensorflow as tf
tf.python.control_flow_ops = tf

data_dir = './data/udacity_data/'
test_data_dir = './data/test_data/'
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
    return newimage


def load_data(X,y,path):
    side_camera_adjustment = 0.2
    i = 0
    f = open(path + '/driving_log.csv')
    for line in f:
        line = line.split(',')
        X.append(read_image(path+'/'+line[0].strip()))
        i += 1
        angle = float(line[3])
        y.append(angle)
        #left camera
        if line[1].strip():
            X.append(read_image(path+'/'+line[1].strip()))
            i += 1
            angle += side_camera_adjustment
            y.append(angle)
        #right camera
        if line[1].strip():
            X.append(read_image(path+'/'+line[2].strip()))
            i += 1
            angle -= side_camera_adjustment
            y.append(angle)
    f.close()
    return X,y
        

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
    
    model.compile('adam', 'mean_squared_error', ['accuracy'])
    return model




if __name__ == "__main__":

    X = []
    y = []
    X,y = load_data(X,y,data_dir)
    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')
    print('X.shape: ',X.shape)
    print('Y.shape: ',y.shape)

    

    mymodel = create_nvidia_model()



    # TODO: train the model
    # samples_per_epoch and batch size must be aligned
    history = mymodel.fit(X,y,nb_epoch=5, shuffle=True,verbose=1,validation_split=0.1)
    
    print(mymodel.summary())
    print(history.history)
    # TODO: check the accuracy
    
    # export the model as required in the assignment
    json_string = mymodel.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
        
    # export the weights as required in the assignment
    mymodel.save_weights('model.h5')


    # test on other pictures
    X_test = []
    y_test = []
    X_test,y_test = load_data(X_test,y_test,test_data_dir)
    X_test = np.array(X_test).astype('float32')
    y_test = np.array(y_test).astype('float32')
    print('X_test.shape: ',X_test.shape)
    print('Y_test.shape: ',y_test.shape)
    
    y_prediction = mymodel.predict(X_test,verbose=1)

    print('predictions: ',y_prediction)
    print('actual: ',y_test)
  
    # Explicitly reap session to avoid an AttributeError sometimes thrown by
    # TensorFlow on shutdown. See:
    # https://github.com/tensorflow/tensorflow/issues/3388
    from keras import backend as K
    K.clear_session()
    
        
