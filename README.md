# CarND-Semantic-Segmentation-Project
## Develop a Fully Convolutional Network to identify and label the areas of an image as drive-able road.

---

[//]: # (Image References)

[image1]: ./support/um_000015.png "Road Image"
[image2]: ./support/um_lane_000015.png "Label for Road Image" 
[image3]: ./support/label_um_000015.png "Road Labeled Image"
[image4]: ./results/random_model_EP-45_LR-0.0001_KP-0.5_BS-15.ckpt_LossCurve.png "Training Loss Curve"
[image5]: ./support/um_000003.png "Road Segment Example"
[image6]: ./support/um_000008.png "Road Segment Example"
[image7]: ./support/um_000014.png "Road Segment Example"
[image8]: ./support/umm_000024.png "Road Segment Example"
[image9]: ./support/umm_000032.png "Road Segment Example"
[image10]: ./support/umm_000040.png "Road Segment Example"
[image11]: ./support/umm_000047.png "Road Segment Example"
[image12]: ./support/umm_000050.png "Road Segment Example"
[image13]: ./support/umm_000066.png "Road Segment Example"
[image14]: ./support/umm_000077.png "Road Segment Example"
[image15]: ./support/uu_000001.png "Road Segment Example"
[image16]: ./support/uu_000005.png "Road Segment Example"
[image17]: ./support/uu_000007.png "Road Segment Example"
[image18]: ./support/uu_000019.png "Road Segment Example"
[image19]: ./results/random_model_EP-50_LR-0.0001_KP-0.5_BS-15.ckpt_LossCurve.png "Training Loss Curve"
[image20]: ./support/50Epoch_uu_000007.png "Over Trained Classify"



## Overview
Utilising transfer learning a pre-trained classifier `VGG16` is converted into a Fully Convolutional Network (FCN). This new model is then trained a series of images in order to perform semantic segmentation on an image. The classification 

---

## Installation steps

To run this code the following downloads are required:

1. Make a project directory `mkdir project_udacity && cd project_udacity`
2. Clone this repository into the project_udacity directory. `https://github.com/nutmas/CarND-Semantic-Segmentation.git`
3. Download dataset: 
      * Download the Kitti Road dataset from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip)
      * Extract the dataset in the data folder inside project folder. This will create the folder data_road with all the training and test images.
      * Example image in dataset:
4. Install Anaconda: `https://www.anaconda.com/distribution/`
5. Build conda environment. `cd CarND-Semantic-Segmentatio-Project\` and launch the script to build the conda environment:
      *  build environment: `conda env create -f environment.yml`
      *  check environments: `conda info -- envs` check carnd-semseg is present
      *  activate environment: `source activate carnd-semseg`
      *  exit conda environment: `. deactivate` back to normal environment.
6. Example images contained in dataset shows image and labelling of road:

      ![alt text][image1]{: width="400px"} ![alt text][image2]{: width="400px"}

---

## Other Important Dependencies

* NVidia GPU with driver capable of running Cuda 8.0
* cmake >= 3.5
* make >= 4.1 
* gcc/g++ >= 5.4

---


## Usage

After completing the installation steps the model can be trained by performing the following step:

1. From terminal window; change to build folder of project `cd ~/project_udacity/CarND-Semantic-Segmentatio-Project/`
2. Activate conda environment: `source activate carnd-semseg`
3. Run the model training: `python main.py`
4. The training will begin showing the epoch number and durations.
5. Upon completion a series of labelled images are stored in the `runs` folder inside the project folder.

**Example of classification of road area in green**

![alt text][image3]{: width="600px"}

---

## Model Documentation

The information received from the simulator required preparation before being passed to the Path Planning.

#### FCN Architecture:

Encoder-Decoder Architecture is employed. Encoder takes the image input and reduces the spatial dimensions using pooling layers; the decoder recovers the object details and spatial dimension and outputs the result. There are some skip layers added to support the decoder with object recovery.

**Encoder:**
VGG-16 is a pre-trained Convolutional Neural Network (CNN). It recognises visual patterns directly from pixel enabling the extraction of features from the image with minimal pre-processing.

**Decoder:**
Layers 3, 4 and 7 of the VGG16 are extracted and used in conjunction with skip layers and upsampled to produce the output.

Fully connected layers 3, 4 and 7 are converted to 1x1 convolutions. 
Layers 3 and 4 have skip connections added.
Layer 7 convolution is upsampled, then added to the layer 4 skip connection, which is then added to the layer 3 skip connection.
An L2 regulariser is added to the convolutions and upsampling to assist in preventing overfitting. The neural net is initialised with random weights 
to remove any bias during training. The RandomNormal initialiser generates tensors with a normal distribution.

**Optimiser:**  
Back-propagation is used to reduce the error during training. Softmax will normalise the probabilities between 0 and 1 and remove the outliers.
Tensorflow function reduce_mean is used to calculate the average error and use the softmax normalisation to understand how far the result is from the ground truth.
The Adam optimiser will update the network weights based on the learning rate and loss result to produce an iterative learning.


#### FCN Hyperparameters:

Hyper Parameters used for final training are:

    * Dropout: (keep_prob) = 0.5
    * Learning Rate: 0.0001
    * EPOCHS: 45
    * Batch size: 15
    * Kernel Initialiser stddev = 0.001
    * Kernel L2 regulariser = 0.0001
    * Number of classes = 2


---

## Results

The model was trained up to 45 EPOCH. The graph shows the training process of loss reducing as number of epoch increases. The final loss of 0.02.

![alt text][image4]{: width="500px"}

The output images showing drive-able area are available at: [Results](./results/random_model_EP-45_LR-0.0001_KP-0.5_BS-15/1554794461.4794238/)
A selection of these images are shown below:


![alt text][image5]{: width="400px"} ![alt text][image6]{: width="400px"} 
![alt text][image7]{: width="400px"} ![alt text][image8]{: width="400px"} 
![alt text][image9]{: width="400px"} ![alt text][image10]{: width="400px"} 
![alt text][image11]{: width="400px"} ![alt text][image12]{: width="400px"} 
![alt text][image13]{: width="400px"} ![alt text][image14]{: width="400px"} 
![alt text][image15]{: width="400px"} ![alt text][image16]{: width="400px"} 
![alt text][image17]{: width="400px"} ![alt text][image18]{: width="400px"} 

---

## Reflection

This project showed how to apply transfer learning in taking an extensively trained network and reusing it to form part of a new network which can produce results quickly.

I experimented with random normal and orthogonal initialisers. The results were very similar for both when fully trained, however the random normal appeared to produce a slightly smoother result and so that was selected.

I tried various number of training epoch to understand the progressional results. I trained with a setting of 50 Epoch. The graph below shows the sudden increase in loss towards the end. The classification is very poor when observing the image output of this model. This was an indicator of over training and resulted in reducing the number of training cycles.

![alt text][image19]{: width="300px"} ![alt text][image20]{: width="550px"} 

Comparing this image from the same the model trained at 50 epoch to the same image above trained at 45 epoch the effect of over training is easy to see in the lack of classification.



---

## License

For License information please see the [LICENSE](./LICENSE) file for details

---

