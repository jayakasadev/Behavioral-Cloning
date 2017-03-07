#**Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator provided by Udacity to collect data of good driving behavior
* Preprocess the data before training
* Build a convolutional neural network using Keras that predicts steering angles from driving images
* Train and validate the model
* Test that the model successfully drives around a test track without leaving the road

#### Repository Contents: 

My project includes the following files:
* model.ipynb containing the code to create and train the model
* model.py contains the code from model.ipynb without any of the illustrations or visualizations
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* Writeup.md which contains a through documentation of my work

#### Technology:
* Anaconda 4.2 for Python 3.5
* Keras
* TensorFlow
* OpenCV

#### Data Collection:
I collected data by driving around the first track multiple times to collect images of certain behaviors:
* Good-Driving Behavior ***(2 laps in each direction)***
* Recovery Driving ***(1 lap)***

Recovery Driving constitutes driving to the edge of each lane and recording myself driving back to the center. I did this for both sides.

Collected data:
* View from 3 different cameras
* Steering angle

All the data was stored in the data directory.

###Model Architecture and Training Strategy

####1. Designing the Model
At first, I tried to base my model on the model used by the [GoogLenet Team](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf). Their model claimed to speedup training by 3-10 times. They did this by makign use of **Inception modules**.  
Unfortunately, their model was 22 layers deep. I was able to build the model, but it was far too big to run on a single 4GB GPU on AWS. I did not realize this right away. It took a few days of playing around and experimenting to figure out the cause of the problem. But, eventually, I realized that there were far too many layers and far too many variables to train with a single GPU.

So, I decided to base my model on the model used by [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) instead.

I later realized that I may still be able to use the GoogLenet structure I created with a generator. But, it was far too complicated a structure and would take too long to setup and run, and I had already gotten the **Nvidia** structure working.



####2. Model Structure:
[![Model Structure](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png)]

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
