# Vehicle Detection and Tracking

## Introduction 

In this project, the goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. Concretely, we

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_exploration.jpg
[image2]: ./output_images/hog_v1.jpg
[image3]: ./output_images/hog_v2.jpg
[image4]: ./output_images/hog_v3.jpg
[image5]: ./output_images/heatmap1.jpg
[image6]: ./output_images/heatmap5.jpg
[image7]: ./output_images/heatmap12.jpg
[image8]: ./output_images/pipeline1.jpg
[image9]: ./output_images/
[image10]: ./output_images/
[video1]: ./output_project_video.mp4
[video2]: ./output_test2.mp4

## Tracking Pipeline
Suppose the video is given. 

For each frame of the video, we run a search for vehicles using a sliding window technique. Wherever the trained classifier (here we trained a Linear SVM classifier) returns a positive detection, we record the position of the window. In some cases, it might detect the same vehicle in overlapping windows or different scales. In the case of overlapping detections, we assign the position of the detection to the centroid of the overlapping windows. We filter out the false positives by determining which detections appear in one frame but not the next. Once we have obtain a high confidence detection, we record how it's centroid is moving from frame to frame and eventually estimate where it will appear in each subsequent frame.

## Dataset
We use [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) in training the linear SVM classifier.

These datasets are comprised of images taken from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.


## Implementation Details
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

In detail, we only search vehicles in a reasonable region, namely, around [0,1280] * [400,656]. For the upper 3/4 part of interested region, we search with small scales. For the lower 3/4 (not 1/4) part of the interested region, we search with larger scales. To decide which scales we are going to use, I first extract (refer to [detectVehicle_final.ipynb](./detectVehicle_final.ipynb) for the code) several representative images and cut two chunks from 'project_video.mp4' as [test images](./test_images) [test video](./test2.mp4). Then, I tried various scales on these images and pick out the most effective ones. 

2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on scales [1.1, 1.4, 1.8, 2.3, 2.6, 2.9] using YCrCb 3-channel HOG features in the feature vector, which provided a nice result.  Here are some example image and video:

![alt text][image8]
and [a test video](./output_test2.mp4)

Please refer to [output images](./output_images) for the selection of scales.
---

### Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_project_video.mp4)


2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]
![alt text][image6]
![alt text][image7]

Since the vehicle detection of a video should have a continuous output, I used the former 25 frames' result to make the final output smooth. 

---

###Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The biggest problem is how to figure out an efficient algorithm. For example, one can search all interested field with every single possible scale, but it is time consuming. Because of this, I collect some representative images and chunks from the test video and pick out the most useful scales. Moreover, each scale only applied to the searching of a specific zone.

* The second problem is how to make the output smooth and reduce false positives. In the submition, I used an average method. Namely, use a global variable to collect former n-frames' heatmap information, and combine them with present heatmap output to obtain a present detection. I think we can also combine previous n-frames' bbox information (with threshold filter) and present ones as the input of a new heatmap with a possibly higher threshold. I guess this method would work too.

* It is very interesting that the algorithm is very sensitive to the setting of scales, which need an explanation.


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
