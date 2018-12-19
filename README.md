# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pipeline_images/dataset_summary_1.png "Example Image"
[image2]: ./pipeline_images/dataset_summary_2.png "Bar Chart"
[image3]: ./pipeline_images/images_preprocessed.png "Grayscaled Image"
[image4]: ./pipeline_images/images_agumented_1.png "Original"
[image5]: ./pipeline_images/images_agumented_2.png "Augmented Image"
[image6]: ./pipeline_images/images_new.png "Real world Images"
[image7]: ./pipeline_images/images_new_predict.png "Real world Images Prediction"
[image8]: ./pipeline_images/image_predict_Speed_limit__30km_h_.png "predict_1"
[image9]: ./pipeline_images/image_predict_Priority_road.png "predict_2"
[image10]: ./pipeline_images/image_predict_Yield.png "predict_3"
[image11]: ./pipeline_images/image_predict_No_entry.png "predict_4"
[image12]: ./pipeline_images/image_predict_Dangerous_curve_to_the_left.png "predict_5"
[image13]: ./pipeline_images/image_predict_Slippery_road.png "predict_6"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/xjtuyanshi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. The first chart randomly picked one of 43 classes images with each class's number of pictures and pecentage % of total number of pictures.
![alt text][image1]
The bar chart showing how the trainning data is unbalanced. Some of them have way many more images than others. This could potentially impact the model accuray.
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
#### Preprocessing
* Grayscale: gray scles could help the image normalization easiler.I converted the color space to YUV space and only pick the Y channel after refering LeCun's paper.
* Contrst Adjust:` cv2.equalizeHist` to imporve the contract of images
* Normalization: applied zero-mean normalization method as it provides way much better results than Udacity's suggested simple normalization method.
Below are the random images after preprocessing:
![alt text][image3]
* Data Augmentation:To add more data to the the data set, I used Keras' `ImageDataGenerator` because we would like to make sure the rotation and zoom in/out should not impact the results.Here is an example of an original image and an augmented image:

![alt text][image4]
![alt text][image5]
The difference between the original data set and the augmented data set is the following 
* zoomed,sheared,rotatated,shifted..


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		  |     Description	        					  | 
|:---------------------:  |:---------------------------------------------:| 
| Input         		  | 32x32x1 Gray image   						  | 
| L1:Convolution 5x5      | 1x1 stride, same padding, outputs 28x28x32	  |
| RELU					  |												  |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x32 				  |
| L2:Convolution 5x5	  | input = 14X14X32 ouput = 10X10X64    		  |
| RELU					  |												  |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x64			          |
| Flatten	              | input = 5x5x64	 ouput = 1600  drop out 0.6   |
| L3:Fully connected	  | input = 1600 ouput = 120  drop out 0.6   	  |
| L4:Fully connected	  | input = 120  ouput = 84  drop out 0.6   	  |
| L5:Fully connected	  | input = 84   ouput = 43 drop out 0.6   	      |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used my ‘Powerful" Nvida GTX 1060 with following parameters:
* Optimizer: Adam Optimizer
* Learning rate: 0.001
* Epoch: 64
* Batch Size:128
* mu: 0
* Sigma: 0.1
* dropout: 0.6

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
The pipeline basically followed example code in the lectures. I didn't change much about it. The training, validation and test code can be found in the ipython notebook.(`evaluate()`).When I tested those parameters, I found larger epouch,batch size and smaller dropout rate helps to get more accurate results.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.992
* test set accuracy of 0.970

If an iterative approach was chosen:
I basiclly just modified leNet structure and just tested different padding size and I found 5X5 is good size to use. Then I referenced some other existing models and articles, I noticed I could add dropout to prevent overfitting. However, initially, the results are still not that good(~85% test accuracy).I noticed I could change the data preprocessing method to improve the accuracy. After using new normalization method, the test accuray rate significantly improved to 94%. However, when I tested it with some real world images, it only has ~20 % accuracy rate, then later I decided to add the data augmentation to improve the robustness of the model. Fortunately, the model test accuracy rate improved to 97% and real world accuracy rate imporved to 50%. A big jump!
If a well known architecture was chosen:
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I tested 16 German traffic signs that I found on the web:

![alt text][image6]

The forth image might be difficult to classify because it seems it is not fully zoomed in.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

![alt text][image7]

The model was able to correctly guess 8 of the 16 traffic signs, which gives an accuracy of 50%. This is way off the  accuracy on the test set. I think one of the reason is that the pictures I collected dosen't follow the standard of test image collection process(some images are zoomed way out,complex backgorund etc.)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 86th cell of the Ipython notebook.

For the first image, the model is 100% sure that this is a stop sign (probability of 1), and the image does contain a 30km/h speed limit sign. The top five soft max probabilities were


![alt text][image8]


For the second image, it doesn't provide the correct results. The 3rd result is the correct one.

![alt text][image9]

For the thrid image, it doesn't provide the correct results at all . None of top5 results matches the actual result.
![alt text][image10]

For the forth image, it doesn't provide the correct results at all . The 3rd result is the correct one.actual result. The differnce between the five probs are not large.Feels like the model is hesitating when make the decision.
![alt text][image11]

For the fifth image, it doesn't provide the correct results  . The 4th result is the correct one.actual one but with nerarly 0 probs.
![alt text][image12]

All other images' top 5 prbs can be found in the folder ./test_images/
