# Behavioral Cloning: Using Deep Learning to Clone Driving Behavior

## Overview
---
### This is Udacity's Self-Driving Car Nanodegree Project.
The goal of this project is to simulate **self-driving** by using behavior cloning. In other words, the "car" is trying to **mimic the driver (me)** without any other aid (ex: pathplanning or preset instructions). In this project, I trained my car by using recorded driving images from a simulator provided by Udacity. In the simulator, I recorded myself driving the simulated car. The data that was captured consists of images (from 3 different camera angles), steering angle, throttle, speed, and braking. I used the images from the 3 camera anngles and the steering angle to train my model. The model was then designed and trained to predict a steering angle in the future, so that it can drive the car autonomously. The  model was evaluated with a track it had never seen before.

### Suggested Reading:
[End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
[Going Deeper with Convolutions](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)
[DeepTesla: End-to-End Learning from Human and Autopilot Driving](http://selfdrivingcars.mit.edu/deeptesla/)

***I used the above research papers to learn more and develop my model***

## Contents
* model.ipynb (notebook used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file


## Dependencies
* Anaconda 4.2
* Keras
* TensorFlow
* OpenCV

# Currently under development
