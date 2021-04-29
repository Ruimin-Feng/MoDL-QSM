 # -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:41:55 2020

@author: frm
"""

# -*- coding: utf-8 -*-


from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
import numpy as np
import os,sys
from scipy.io import loadmat
from scipy.io import savemat
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from model.MoDL_QSM import define_generator

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def data_read(data_orders, pathname):
    """Load the training datasets """
    read_data = []
    for i in data_orders:
        data_temp = loadmat(pathname +str(i) + '.mat')
        read_data.append(data_temp)
    print(len(read_data))
    return read_data

        
def generate_arrays_from_file(data,D_input,lines,batch_size,Normdata,input_patch_shape=[48,48,48],output_patch_shape=[48,48,48]):
    n=len(lines)
    i=0
    CosTrnMean=Normdata['CosTrnMean']
    CosTrnStd=Normdata['CosTrnStd']
    CosTrnMean=CosTrnMean[np.newaxis,np.newaxis,:,:]
    CosTrnMean=np.tile(CosTrnMean,(48,48,48,1))
    CosTrnStd=CosTrnStd[np.newaxis,np.newaxis,:,:]
    CosTrnStd=np.tile(CosTrnStd,(48,48,48,1))
    while 1:
        batch_X_input=[]
        batch_mask=[]
        batch_D_input=[]
        batch_Y=[]
        
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            #generate inputs and labels
            temp=lines[i]
            subject=temp[0]
            batch_data=data[subject-1]         
                                   
           
            x_input=batch_data['x_input'][int(temp[1]-input_patch_shape[0]/2):int(temp[1]+input_patch_shape[0]/2),int(temp[2]-input_patch_shape[1]/2):int(temp[2]+input_patch_shape[1]/2),int(temp[3]-input_patch_shape[2]/2):int(temp[3]+input_patch_shape[2]/2)]
            mask=batch_data['msk'][int(temp[1]-input_patch_shape[0]/2):int(temp[1]+input_patch_shape[0]/2),int(temp[2]-input_patch_shape[1]/2):int(temp[2]+input_patch_shape[1]/2),int(temp[3]-input_patch_shape[2]/2):int(temp[3]+input_patch_shape[2]/2)]
            msk=np.expand_dims(mask,axis=-1)
            msk=np.tile(msk,(1,1,1,2))
            Ydata=batch_data['labels'][int(temp[1]-output_patch_shape[0]/2):int(temp[1]+output_patch_shape[0]/2),int(temp[2]-output_patch_shape[1]/2):int(temp[2]+output_patch_shape[1]/2),int(temp[3]-output_patch_shape[2]/2):int(temp[3]+output_patch_shape[2]/2),:]
            
            Ydata=(Ydata-CosTrnMean)/CosTrnStd*msk           

            batch_X_temp = np.expand_dims(x_input,axis=3)
            batch_mask_temp = np.expand_dims(mask, axis=3)
            batch_D_input_temp=np.expand_dims(D_input, axis=3)
            batch_Y_temp = Ydata
           

            batch_X_input.append(batch_X_temp)
            batch_mask.append(batch_mask_temp)
            batch_D_input.append(batch_D_input_temp)        
            batch_Y.append(batch_Y_temp)
         
            
            i=(i+1)%n

        yield [np.array(batch_X_input),np.array(batch_mask),np.array(batch_D_input)],np.array(batch_Y)

        
             
    
        
if __name__=="__main__":
    log_dir="../logs/ "
    Normdata=loadmat('../NormFactor.mat')
    model=define_generator(3,1,Normdata,matrix_size=[210,224,160],input_shape=[48,48,48],output_size=[48,48,48])
    model.summary()
    
    """load training data and dipole kernel"""
    data = data_read(np.arange(1,91), '../train_data/')
    D_data=loadmat('../train_data/D.mat')       
    D_input=D_data['D']
    print('ok')
    
    """This is the patch index: [sub,x,y,z]  """
    lines_data=loadmat('../lines.mat')   
    lines_array=lines_data['index']
    lines=lines_array.tolist()          #############
    num=len(lines) 
    
    
    checkpoint_period=ModelCheckpoint(log_dir+'ep{epoch:03d}-loss{loss:.3f}.h5',
                                      monitor='val_loss',
                                       save_weights_only=True,
                                       save_best_only=True,
                                       period=1
                                        )
  
    #learning rate
    reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=3,
                                verbose=1) 
    #if need early stop
    early_stopping=EarlyStopping(monitor='val_loss',
                                 min_delta=0,
                                 patience=20,
                                 verbose=1)
    tbCallBack=TensorBoard(log_dir="../logs/tnsboard")  

    #loss
    model.compile(loss='mae',
                        optimizer=Adam(lr=1e-3))     
   #                    
    batch_size=2
    num_train=np.int(0.8*num)
    num_val=np.int(num-num_train)
    np.random.shuffle(lines)
    print('Train on {} samples,val on {} samples,with batch size {}.'.format(num_train,num_val,batch_size))
   
   #start train
    model.fit_generator(generate_arrays_from_file(data,D_input,lines[:num_train],batch_size,Normdata),
                       steps_per_epoch=3000,
                       validation_data=generate_arrays_from_file(data,D_input,lines[num_train:],batch_size,Normdata),
                       validation_steps=300,
                       epochs=40,
                       initial_epoch=0,
                       callbacks=[checkpoint_period,reduce_lr,tbCallBack,early_stopping])
    model.save_weights(log_dir+'last.h5')
   
    
