# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.jpg "Center lane driving"
[image3]: ./examples/r2c0.jpg "Recovery Image(Right to Center 1/2)"
[image4]: ./examples/r2c1.jpg "Recovery Image(Right to Center 2/2)"
[image5]: ./examples/l2c0.jpg "Recovery Image(Left to Center 1/2)"
[image6]: ./examples/l2c1.jpg "Normal Image(Left to Center 2/2)"
[image7]: ./examples/flipped.jpg "Flipped Image"
[image8]: ./examples/figure_epoch5.png "Loss for epochs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 74-91) 

The model includes RELU layers to introduce nonlinearity (code line 78-82), and the data is normalized in the model using a Keras lambda layer (code line 76). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 84). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 118). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving(2 laps for clock-wise and 2 laps for counter clock-wise), recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to capture center lane driving.

My first step was to use a convolution neural network model similar to the model proposed by Nvidia.I thought this model might be appropriate because it have five Convolution layers which may have a good performance in extracting the characteristic of lanes and ledges. And the rest 4 fc layer  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add some modified the model so that 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track(e.g. at the right corner after crossing the bridge). To improve the driving behavior in these cases, I train the model with more data including clock-wise laps and recovery laps.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-90) consisted of a convolution neural network with the following layers and layer sizes.
Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the middle. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would  For example, here is an image that has then been flipped:
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by normalization and mean centering.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by decrease of the validation loss for almost all of the epochs.I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image8]

