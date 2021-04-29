# -*- coding: utf-8 -*-
"""
This file contains the main test code for two Prisma data, one MS data and QSM 2016 challenge data
Run this code and the results will be in the same directory as the input data 

Created on Thu May 21 14:39:20 2020

@author: frm
"""
import numpy as np
from test_tools import *
from scipy.io import loadmat
from scipy.io import savemat
model_dir='../logs/last.h5'

"""prisma test data subject1""" 
data_path='../data/Prisma_data/sub1/' 
test_data=loadmat(data_path+'test_data.mat')
phi_data = test_data['phi']
mask_data = test_data['mask']
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1],True)
save_nii(Y_data,data_path+'MoDL_QSM_output.nii.gz',[1,1,1])

     
"""prisma test data subject2""" 
data_path='../data/Prisma_data/sub2/' 
test_data=loadmat(data_path+'test_data.mat')
phi_data = test_data['phi']
mask_data = test_data['mask']
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1],True)
save_nii(Y_data,data_path+'MoDL_QSM_output.nii.gz',[1,1,1])


"""MS test data""" 
data_path='../data/MS_data/' 
test_data=loadmat(data_path+'test_data.mat')
phi_data = test_data['phi']
mask_data = test_data['mask']
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1],True)
save_nii(Y_data,data_path+'MoDL_QSM_output.nii.gz',[1,1,1])


"""QSM 2016 challenge test data"""
data_path='../data/QSM_challenge_data/' 
test_data=loadmat(data_path+'test_data.mat')
phi_data = test_data['phi']
mask_data = test_data['mask']
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1],True)
save_nii(Y_data,data_path+'MoDL_QSM_output.nii.gz',[1,1,1])

"""Hemorrhage test data"""
data_path='../data/Hemorrhage/' 
test_data=loadmat(data_path+'test_data.mat')
phi_data = test_data['phi']
mask_data = test_data['mask']
Y_data=model_test(model_dir,phi_data,mask_data,[0.8594,0.8594,2],[0,0,1],True)
save_nii(Y_data,data_path+'MoDL_QSM_output.nii.gz',[0.8594,0.8594,2])


