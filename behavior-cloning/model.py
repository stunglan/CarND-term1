# The script used to create and train the model.

# check: https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
#  

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU


from PIL import Image
import cv2
import numpy as np
import json
import argparse
import time
import os
import math
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.python.control_flow_ops = tf

rows = 20
cols = 40
channels = 3


data_dir = './data/myno1/'
data_dir = './data/mattew_data/'
data_dir = './data/udacity_data/'

test_data_dir = './data/test_data/'
 

def random_shear(image,steering,prob=0.8):
    # tnx: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.yozfr1waw
    #print('steering: ',steering)
    if np.random.random_sample() > prob and steering < 0.01:
        rows = image.shape[0]
        cols = image.shape[1]
        shear_range = math.ceil(cols/3)
        dx = np.random.randint(-shear_range,shear_range+1)
        random_point = [cols/2+dx,rows/2]
        pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
        pts2 = np.float32([[0,rows],[cols,rows],random_point])
        dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 3.0
        M = cv2.getAffineTransform(pts1,pts2)
        image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
        steering +=dsteering
    return image,steering


def random_flip(im,angle,prob=0.5):
    if np.random.random_sample() > prob:
        return np.fliplr(im),angle * -1.0
    else:
        return im,angle

    

def read_image(filename):
    newimage = Image.open(filename)
    image_array = np.asarray(newimage)
    return image_array

def process_image(image_array,y):
    # crop the image
    top = math.ceil(image_array.shape[0]*0.30)
    bot = math.ceil(image_array.shape[0]-image_array.shape[0]*0.1)
    image_array = image_array[top:bot, :]

    # color the image
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)

    # randomly shear the image
    image_array,y = random_shear(image_array,y)

    # resize the image
    image_array = cv2.resize(image_array,(cols,rows))
    # flip randomly the image

    image_array,y = random_flip(image_array,y)

    return image_array,y

def load_data(X,y,path):
    print('start loading data...')
    t = time.process_time()
    side_camera_adjustment = .23
    f = open(path + '/driving_log.csv')
    for line in f:
        line = line.split(',')

        # center camera
        im = read_image(path+'/'+line[0].strip()) # center image
        angle = float(line[3])
        X.append(im)
        y.append(angle)

        #left camera
        if line[1].strip():
            im = read_image(path+'/'+line[1].strip())
            angle = angle + side_camera_adjustment # if this close, steer to the right
            X.append(im)
            y.append(angle)
            
        #right camera
        if line[2].strip():
            im = read_image(path+'/'+line[2].strip())
            angle = angle - side_camera_adjustment # if this close, steer to the right
            X.append(im)
            y.append(angle)
            
    elapsed_time = time.process_time() - t
    print('end loading data ---~> {}'.format(elapsed_time))
    print('mean y before augumentation ', np.mean(y))

    for i in range(len(y)):
        # augument the image
        X[i],y[i] = process_image(X[i],y[i])

    print('mean y after augumentation ', np.mean(y))
    f.close()
    return X,y


# generate function for model.fit_generator
def generate_batch(X, y, batch_size):
    x_batch = np.zeros((batch_size, rows, cols, channels))
    y_batch = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            i_line = np.random.randint(len(y))
            x_batch[i] = X[i_line]
            y_batch[i] = y[i_line]
        yield x_batch, y_batch
                
def create_nvidia_model():
    """The nividia model 
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()
    # Use a lambda layer to normalize the input data
    model.add(Lambda(
        lambda x: x/127.5 - 1.,
        input_shape=(rows, cols, channels),
        output_shape=(rows, cols, channels))
    )

    # Several convolutional layers, each followed by ELU activation
    # ELU http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf
    act = ELU
    
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(act())

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(act())
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(act())
    
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(act())

    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(act())
    
    model.add(Flatten())
    model.add(act())

    model.add(Dropout(.5))
    model.add(act())

    model.add(Dense(100))
    model.add(act())

    model.add(Dense(50))
    model.add(act())

    model.add(Dropout(.5))
    model.add(act())
    
    
    # Fully connected layer with one output dimension (representing the steering angle).
    model.add(Dense(1))
    return model


if __name__ == "__main__":

    # command line argument
    parser = argparse.ArgumentParser()
    # how much output
    parser.add_argument('-v','--verbosity',default=1,help='how much output',type=int, choices=[0, 1, 2])
    # add new data
    parser.add_argument('-a','--add',help='add a data directory')
    args = parser.parse_args()

    # test if the data directory exist
    if args.add:
        data_dir = args.add
        if not os.path.isdir(data_dir):
            print('directory does not exist {}'.format(data_dir))
    print('loading data from {}'.format(data_dir))
    
        
    # load the data 
    X = []
    y = []
    X,y = load_data(X,y,data_dir)
    print('type X: ',type(X))
    print('is list X: ',isinstance(X,list))
    
    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')
    if args.verbosity >= 1:
        print('X.shape: ',X.shape)
        print('Y.shape: ',y.shape)



    # split the training data and the validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        
    # build the model
    mymodel = create_nvidia_model()

    # load weigths if we add data
    if args.add:
        print('adding to existing weights')
        mymodel.load_weights('model.h5')
    
    # compile the model
    adam = Adam(lr=0.001)
    mymodel.compile(loss='mean_squared_error',optimizer=adam)
    # train the model
    #history = mymodel.fit(X,y,nb_epoch=50,shuffle=True, batch_size=256, verbose=1,validation_split=0.1)
    history = mymodel.fit_generator(generate_batch(X_train, y_train, 250),
                                  samples_per_epoch=250 * 100,
                                  nb_epoch=50, verbose=1,
                                  validation_data=generate_batch(X_val, y_val, 250),
                                  nb_val_samples=len(y_val)
      )
    # display the model
    if args.verbosity >= 2:
        print(mymodel.summary())
        print(history.history)
    
    
    # export the model and weight as required in the assignment
    if args.verbosity >= 1:
        print('Dumping model and weights...') 
    with open('model.json', 'w') as outfile:
        json.dump(mymodel.to_json(), outfile)
    mymodel.save_weights('model.h5')


    # test on other pictures
    if args.verbosity >= 1:
        print('Comparing 3 frames...') 

        X_test = []
        y_test = []
        X_test,y_test = load_data(X_test,y_test,test_data_dir)
        X_test = np.array(X_test).astype('float32')
        y_test = np.array(y_test).astype('float32')
        print('X_test.shape: ',X_test.shape)
        print('Y_test.shape: ',y_test.shape)
        
        y_prediction = mymodel.predict(X_test,verbose=1)
        
        for i in range(len(y_prediction)):
            print('predictions vs actual: {}: {:+5.3f} should be {:+5.3f}'.format(i, y_prediction[i][0], y_test[i]))
            
            
    # Explicitly reap session to avoid an AttributeError sometimes thrown by
    # TensorFlow on shutdown. See:
    # https://github.com/tensorflow/tensorflow/issues/3388
    from keras import backend as K
    K.clear_session()
    
        
