# MoDL-QSM
Source codes and trained networks described in the paper: [MoDL-QSM: Model-based Deep Learning for Quantitative Susceptibility Mapping](https://www.sciencedirect.com/science/article/pii/S1053811921006522) 

MoDL-QSM was proposed by Ruimin Feng and Dr. Hongjiang Wei. It reconstructs high quality STI (Susceptibility Tensor Imaging) component χ_33 map and the field induced by χ_13 and χ_23 terms from the tissue phase.

## Environmental Requirements:  

Python 3.6  
Tensorflow 1.15.0  
Keras 2.2.5  

## Files descriptions:  
MoDL-QSM contains the following folders:  

data: It provides four types of test data: two healthy data from Siemens Prisma scanner, multiple sclerosis data, 2016 QSM Challenge data, and hemorrhage data.

logs/last.h5: A file that contains the weights of the trained model

model/MoDL_QSM.py : This file contains the functions to create the model-based convolutional neural network proposed in our paper

test: It contains test_tools.py and test_demo.py.
test_tools.py offers some supporting functions for network testing such as image patch stitching, dipole kernel generation, etc. test_demo.py shows how to perform network testing with data from the "data" folder

train: It contains train_lines.py.
train_gen.py: This is the code for network training

NormFactor.mat: The mean and standard deviation of our training dataset for input normalization.

## Usage  
### Test  
You can run test_demo.py directly to test the network performance on the provided data. The results will be in the same directory as the input data  
For test on your own data. You can use "model_test" function as shown in test_demo.py files  

### train  
If you want to train MoDL-QSM by yourself. train_lines.py can be used as a reference.
