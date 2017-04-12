# Vehicle Detection and Tracking

## Introduction 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_exploration.jpg
[image2]: ./output_images/hog_v1.jpg
[image3]: ./output_images/hog_v2.jpg
[image4]: ./output_images/hog_v3.jpg
[image5]: ./output_images/
[image6]: ./output_images/
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in [HOG_classifier_v3.ipynb](./HOG_classifier_v3.ipynb).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

Here is two examples using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

![alt text][image3]
![alt text][image4]

2. Explain how you settled on your final choice of HOG parameters.

Recall that HOG divides an input image into several small pieces. Then for each piece, it calculates the gradient of variation in a given number of orientations. Simply put, HOG describes the variation of colors. The greater the variation, the greater the gradient. Therefore, the less the number of pixels per cells (and other parameters), more general the data, and the more, more specific. 

I tried various combinations of parameters and found that 8 orientations, and 8 pixels per cell are enough to identify a car. 

3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the corresponding function in scikit-learn. 

For above fixed orientation, pixel per cell and cell per block parameters, I tried further several combinations of colorspace and hog_channel. When using 'YCrCb' as colorspace and choose 'ALL' hog_channel, I obtained the highest test accuracy (0.9775). It is statistically higher than other combinations (between 0.93-0.96) I've tried according to the rule of 30.

This classifier and the scaler were saved using pickle library, which will be used later in the vehicle detection. 

### Sliding Window Search

1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

There are at least two ways to implement a sliding window search. Either use a size-fixed window to silde in different scaled image, or use different sized window to slide in the same image. I tried both methods. In this submission, only the first method is considered. 

In detail, we only search vehicles in a reasonable region, namely, around [0,1280] * [400,656]. For the upper 2/3 part of interested region, we search with small scales. For the lower 2/3 (not 1/3) part of the interested region, we search with larger scales. To decide which scales we are going to use, I first extract (see the bottom of [detectVehicle.ipynb](./detectVehicle_v5.ipynb) for the code) several representative images from 'project_video.mp4' as [test images](./test_images). Then, I try various scales on these images and pick out the most effective ones. 

2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on scales [1.1, 1.4, 1.5, 1.6, 1.7, 1.8, 2.3, 2.3, 2.5, 2.8, 2.9] using YCrCb 3-channel HOG features in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]

---

### Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I met here is time costing. I am looking for a more efficient method. The second problem is 


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
