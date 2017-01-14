## P3 project in Self-Driving Car nanodegree


### Problem description
If everything went correctly, you should see the following in the directory you selected:

1. IMG folder - this folder contains all the frames of your driving.
1. driving_log.csv - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll mainly be using the steering angle.

#### Training Your Network
Now that you have training data, it’s time to build and train your network!

Use Keras to train a network to do the following:

1. Take in an image from the center camera of the car. This is the input to your neural network.
1. Output a new steering angle for the car.


#### Validating Your Network
You can validate your model by launching the simulator and entering autonomous mode.

The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

Set up your development environment with the CarND Starter Kit.
1. Download drive.py.
1. Run the server.

python drive.py model.json
If you're using Docker for this project: docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starer-kit python drive.py model.json or docker run -it --rm -p 4567:4567 -v ${pwd}:/src udacity/carnd-term1-starer-kit python drive.py model.json. Port 4567 is used by the simulator to communicate.

Once the model is up and running in drive.py, you should see the car move around (and hopefully not off) the track!