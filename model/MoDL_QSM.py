# -*- coding: utf-8 -*-
"""
This code will create the generative and  adversarial neural network described in our following paper
MoG-QSM: Model-based Generative Adversarial Deep Learning Network for Quantitative Susceptibility Mapping

Created on Mon Jul 20 14:35:52 2020

@author: frm
"""

from keras.models import Model
from keras.layers import Input,BatchNormalization,Dropout,Lambda,Add,LeakyReLU,Multiply,Activation,Flatten,Layer
from keras.layers import Conv3D
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import os

weight_decay=0.0005
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def initial_conv(x,filters,kernel_size,strides=(1,1,1),padding='same'):
    """
    This function creates a convolution layer followed by ReLU
    """
    x=Conv3D(filters,kernel_size,strides=strides,padding=padding,
             W_regularizer=l2(weight_decay),
             use_bias=False,
             kernel_initializer='he_uniform')(x)
    
    x=Activation('relu')(x)
    return x


def Res_block(x, k=1, dropout=0.0):
    """
    This function creates a ResBlock
    """
    
    init=x
    
    x = Conv3D(16*k,(3,3,3),padding='same',
               W_regularizer=l2(weight_decay),
               use_bias=False,
               kernel_initializer='he_uniform')(x)
       
    x = BatchNormalization(axis=4)(x)
    x=Activation('relu')(x)    
    
    if dropout > 0.0: x=Dropout(dropout)(x)
    
    x = Conv3D(16*k,(3,3,3),padding='same',
               W_regularizer=l2(weight_decay),
               use_bias=False,
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(axis=4)(x)
    m = Add()([init,x])
    m=Activation('relu')(m)
    
    return m

def pad_tensor(x,ind1,ind2,ind3,matrix_size):
    """
    This function will pad the output patches to match the size of dipole kernel
    """
    paddings = tf.constant([[0,0],[int((matrix_size[0]-ind1)/2), int((matrix_size[0]-ind1)/2)], [int((matrix_size[1]-ind2)/2), int((matrix_size[1]-ind2)/2)],[int((matrix_size[2]-ind3)/2),int((matrix_size[2]-ind3)/2)],[0,0]])
    px=tf.pad(x,paddings,'CONSTANT')
    return px

    
def AH_op(x,ind1,ind2,ind3,matrix_size):
    """
    This function is the A^H operator as described in paper
    """
    phase=x[0]
    D=x[1]

    #
    phase=tf.dtypes.cast(phase,tf.complex64)
    D=tf.dtypes.cast(D,tf.complex64)
    
    
    phase=tf.transpose(phase,perm=[0,4,1,2,3])
    D=tf.transpose(D,perm=[0,4,1,2,3])
    
    #scaling factor
    SF=np.sqrt(matrix_size[0]*matrix_size[1]*matrix_size[2])
    SF=tf.dtypes.cast(tf.convert_to_tensor(SF),tf.complex64)
    
    #A_H_A
    ksp_phase=tf.signal.fft3d(phase)/SF
    ty=tf.signal.ifft3d(tf.multiply(D,ksp_phase))*SF
    ty=tf.transpose(ty,perm=[0,2,3,4,1])
    phase=tf.transpose(phase,perm=[0,2,3,4,1])
    ty=tf.concat([ty,phase],axis=-1)
    #cut to the original size
    y=ty[:,int((matrix_size[0]-ind1)/2):int((matrix_size[0]-ind1)/2)+ind1,int((matrix_size[1]-ind2)/2):int((matrix_size[1]-ind2)/2)+ind2,int((matrix_size[2]-ind3)/2):int((matrix_size[2]-ind3)/2)+ind3,:]   
    y=tf.dtypes.cast(y,tf.float32)
    return y

def A_op(x,ind1,ind2,ind3,matrix_size):
    """
    This function is the A operator as described in paper
    """
    sus=x[0]
    D=x[1]
    
    #dipole kernel
    sus=tf.dtypes.cast(sus,tf.complex64)
    D=tf.dtypes.cast(D,tf.complex64)

    sus0=sus[:,:,:,:,0]
    sus1=sus[:,:,:,:,1]
 
    #scaling factor
    SF=np.sqrt(matrix_size[0]*matrix_size[1]*matrix_size[2])
    SF=tf.dtypes.cast(tf.convert_to_tensor(SF),tf.complex64)
    #A
    ksp_sus0=tf.signal.fft3d(sus0)/SF
    ksp_sus1=tf.signal.fft3d(sus1)/SF

    ty=tf.signal.ifft3d(tf.multiply(D[:,:,:,:,0],ksp_sus0)+ksp_sus1)*SF
    #cut to the original size
    y=ty[:,int((matrix_size[0]-ind1)/2):int((matrix_size[0]-ind1)/2)+ind1,int((matrix_size[1]-ind2)/2):int((matrix_size[1]-ind2)/2)+ind2,int((matrix_size[2]-ind3)/2):int((matrix_size[2]-ind3)/2)+ind3]   
    y=tf.expand_dims(y,axis=-1)
    y=tf.dtypes.cast(y,tf.float32)
    return y
    

def term2(inputs,ind1,ind2,ind3,matrix_size):
    """
    This function performs the term: -t_k A^H A x^k-1 as described in paper  
    """
    sus=inputs[0]
    D=inputs[1]
    alpha=inputs[2]
    pad_sus=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})(sus)
    A_sus=Lambda(A_op,output_shape=(ind1,ind2,ind3,1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})([pad_sus,D])
    pad_sus=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})(A_sus)
    AH_sus=Lambda(AH_op,output_shape=(ind1,ind2,ind3,2),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})([pad_sus,D])
   
    weight2=Lambda(lambda y:-y[1]*y[0])
    x=weight2([AH_sus,alpha])
    return x

def term1(y,ind1,ind2,ind3,matrix_size):
    """
    This function performs the term: t_k A^H y as described in paper 
    """
    pad_phase=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})(y[0])
    out_y=Lambda(AH_op,output_shape=(ind1,ind2,ind3,2),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})([pad_phase,y[1]])
    return out_y
    
def init(x):
    """
    This function creates a zero keras tensor of the same size as x
    """
    return tf.zeros_like(x)
        
def My_init(shape, dtype='float32',name=None):
    """
    This function creates the learnable step size in gradient descent    
    """

    value =4.

    return K.variable(value, name=name)



class MyLayer(Layer):
    """
    This function creates the custom layer in Keras
    """
    
    def __init__(self,**kwargs):
     
        super(MyLayer,self).__init__(**kwargs)
        
    def build(self,input_layer):
        self.step=self.add_weight(name='step',
                                    shape=(1,1),
                                    initializer=My_init,
                                    trainable=True)
        super(MyLayer,self).build(input_layer)
        
    def call(self, input_layer):
        return self.step*tf.ones(tf.shape(input_layer))



      
        
def define_generator(num_iter,is_train,Normdata,matrix_size=[210,224,160],input_shape=[48,48,48],output_size=[48,48,48]):
    """
    This function is called to create the generator model
    """
    
    CosTrnMean=Normdata['CosTrnMean']
    CosTrnStd=Normdata['CosTrnStd']
    CosTrnMean=CosTrnMean[np.newaxis,np.newaxis,:,:]
    CosTrnMean=np.tile(CosTrnMean,(input_shape[0],input_shape[1],input_shape[2],1))
    CosTrnStd=CosTrnStd[np.newaxis,np.newaxis,:,:]
    CosTrnStd=np.tile(CosTrnStd,(input_shape[0],input_shape[1],input_shape[2],1))
        
    #######G block###########
    init_input=Input(shape=input_shape+[2])
    conv1=initial_conv(init_input,32,(3,3,3))
    print(conv1.shape)
    wide_res1=Res_block(conv1,k=2,dropout=0.5)
    wide_res2=Res_block(wide_res1,k=2,dropout=0.5)
    wide_res3=Res_block(wide_res2,k=2,dropout=0.5)
    wide_res4=Res_block(wide_res3,k=2,dropout=0.5)
    wide_res5=Res_block(wide_res4,k=2,dropout=0.5)
    wide_res6=Res_block(wide_res5,k=2,dropout=0.5)
    wide_res7=Res_block(wide_res6,k=2,dropout=0.5)
    wide_res8=Res_block(wide_res7,k=2,dropout=0.5)
        
    conv2=initial_conv(wide_res8,32,(1,1,1))
    print(conv2.shape)
    conv3=initial_conv(conv2, 32, (1,1,1))
    output=Conv3D(filters=2,kernel_size=(1,1,1),strides=(1,1,1),padding='same')(conv3)
        
    print(output.shape)
    basic_model=Model(init_input,output)
    
    ########P(physical) block#########
    y_init=Input(shape=input_shape+[1])    #input phase
    mask=Input(shape=input_shape+[1])      #input voxel_size
    msk=Lambda(lambda x:tf.tile(x,(1,1,1,1,2)))(mask)
    D_input=Input(shape=matrix_size+[1])       #input dipole kernel
    size=tf.keras.backend.int_shape(y_init) 
       
    y_input=Lambda(term1,output_shape=(size[1],size[2],size[3],2),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size})([y_init,D_input])
    Alpha= MyLayer()(y_input)
    y_input=Multiply()([y_input,Alpha])

    #iterative
    for i in range(num_iter):
        if i==0:
            layer_input=y_input
        else:
            term_output=Lambda(term2,output_shape=(size[1],size[2],size[3],2),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size})([x_output,D_input,Alpha]) #lamda_9  #lamda_20 #lamda_31
            layer_output=Add()([x_output,term_output])    #add_9  add_11 add_13
            layer_input=Add()([layer_output,y_input])    #add_10    add_12  add_14
                    
        layer_input=Lambda(lambda x: (x-CosTrnMean)/CosTrnStd)(layer_input)
        layer_input=Multiply()([layer_input,msk])
        fx_output=basic_model(layer_input)   #model_1  
        fx_output=Multiply()([fx_output,msk])
        x_output=Lambda(lambda x: x*CosTrnStd+CosTrnMean)(fx_output)
        
    
    if is_train:       
        model=Model([y_init,mask,D_input],fx_output)
        return model
    else:
        x_output=Multiply()([x_output,msk])
        model=Model([y_init,mask,D_input],x_output)
        return model






