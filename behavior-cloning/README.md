;;; README.md --- us

;;; Commentary:
;;

# P3 project in Self-Driving Car nanodegree

## Explanation of the structure of my network and training approach.

###Network
I fairly early chose the NVIDIA network for training my network, which I modified slightly. I also tried a VGG16, but that did not immediate improve the performance so I stuck to the NVIDIA network. The original network has 9 layers including a normalisation layer, 5 convolutional layers - three 5x5 kernels with 24,36 and 48 output depths, and two 3x3 layers with both 64 output depth – flattened the model and then lastly 2 fully connected layers. I modified the networ## Explanation of the structure of my network and training approach.

### Data capture
I used the provided dataset, including the left and right camera. In addition, I drove the simulator myself with a MacBook and mouse, I collected two dataset with two rounds. I loaded the datasets by storing and loading the tensors.

###Network
I fairly early chose the [NVIDIA] (https://arxiv.org/pdf/1412.6980v8.pdf)  network for training my network, which I modified slightly. I also tried a VGG16, but that did not immediate improve the performance so I stuck to the NVIDIA network. The original network has 9 layers including a normalisation layer, 5 convolutional layers - three 5x5 kernels with 24,36 and 48 output depths, and two 3x3 layers with both 64-output depth – flattened the model and then lastly 2 fully connected layers. I modified the network by adding a dropout layer after the flattening and the second fully connected layer, both with a probability of 0.5.

### Image pre-processing
The steering angle for the left and right camera is adjusted slightly to adjust for their positioning. I then crop the image 30% on the top and 10% at the bottom. Then changed the colormapping from RGB to YUV. Most pictures are when the road is straight ahead, to increase the training set when steering is needed I shear the images and the steering angle accordingly, I do this for approximately 50% of the images. I also flip circa 50% left to right to avoid a to biased training for left turns. I resize the picture down to a 20 rows by 40 columns picture.

### Method
I struggled quite a bit. Firstly, I tried using only my data, but the dataset that I got was poor, and it was hard to drive the car. I then decided to use the dataset from the course material to ensure that I had some proper data.

During development, I used a simplified network model, and promptly forgot that I simplified it. I then struggled a with fine tuning the pre-processing, reading up on what other students did and wondered why they got the car to run, and I did not. I finally got the car around the track, but very wobbly. And when I was cleaning up the code I discovered that was using the simplified network model, some of the jerkiness disappeared when I reintroduced the full network. I also tried a VGG16 network, without a pre-trained model, this did not perform better so I did not pursue that more.

In the pre-processing I struggled most with the shearing, but I had help from fellow students to create a proper routine for that. I also used jupyter to see what the pre-processing did for the pictures.

I did not collect any statistics about the data, steering angles, lightning and such, and could probably succeed faster had I done this.

### Network summary
‘’’
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 20, 40, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 10, 20, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 10, 20, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 5, 10, 36)     21636       elu_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 5, 10, 36)     0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 3, 5, 48)      43248       elu_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 3, 5, 48)      0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 2, 3, 64)      27712       elu_3[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 2, 3, 64)      0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 2, 64)      36928       elu_4[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 2, 64)      0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 128)           0           elu_5[0][0]
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 128)           0           flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 128)           0           elu_6[0][0]
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 128)           0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           12900       elu_7[0][0]
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_8[0][0]
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           elu_9[0][0]
____________________________________________________________________________________________________
elu_10 (ELU)                     (None, 50)            0           dropout_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             51          elu_10[0][0]
====================================================================================================
Total params: 149,349
Trainable params: 149,349
Non-trainable params: 0
‘’’




###list of articles:
* [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION] (https://arxiv.org/pdf/1412.6980v8.pdf)
* [End-to-End Deep Learning for Self-Driving Cars] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
* [Behavioral Cloning — make a car drive like yourself] (https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.yozfr1waw)

(provide 'README)

;;; README.md ends here
