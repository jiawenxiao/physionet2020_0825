#!/usr/bin/env python
import numpy as np, os, sys
import joblib
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_features(data,header_data): 
    set_length=5000
    data_num = np.zeros((1,8,set_length))
    data_external= np.zeros((1,3))
    
    leads=[0,1,6,7,8,9,10,11]
    data=data[leads,:]
    
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0][0]
              
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    tmp_length= int(tmp_hea[3])
    gain_lead = np.zeros(num_leads)
    
    if ptID[0]=='Q':
        source=0
    elif ptID[0]=='S':
        source=1
    elif ptID[0]=='I':
        source=2
    elif ptID[0]=='A':
        source=3
    elif ptID[0]=='E':
        source=4
    elif ptID[0]=='H':
        source=5
    else:
        source=6
                     
    if sample_Fs==1000:   
        rs_idx=range(0,len(data[0]),2)   
        data=data[:,rs_idx]  

    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])
    
    for i,lines in enumerate(header_data):        
        if lines.startswith('#Age'):
            tmp_age = lines.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
            age=age/100 
        elif lines.startswith('#Sex'):
            tmp_sex = lines.split(': ')[1]
            if tmp_sex.strip()=='Female'  or tmp_sex.strip()=='F':
                sex =1
            else:
                sex=0
                
                
    ##添加去除基线漂移
    tmp=data.mean(axis=1).reshape(8,1)
    tmp_mean=np.repeat(tmp,data.shape[1],axis=1)
    data=data-tmp_mean  
            
    if tmp_length>= set_length:
        data_num[:,:,:] = data[:,: set_length]*gain_lead[0]
    else:
        data_num[:,:,:tmp_length] = data*gain_lead[0]
        
    data_num= data_num.reshape(1,8,-1)   
    
    data_external[:,0] =age 
    data_external[:,1] =sex  
    data_external[:,2] =source
    
    return data_num, data_external

def load_12ECG_model(input_directory):
    # load the model from disk 
    f_out='resnet_0825.pkl'
    filename = os.path.join(input_directory,f_out)
    loaded_model = torch.load(filename,map_location=device)
    return loaded_model


def run_12ECG_classifier(data,header_data,model):       
    classes=['270492004','164889003','164890007','426627000','713427006','713426002','445118002','39732003',
                  '164909002','251146004','10370003','284470004','164947007','111975006',
                  '47665007','59118001','427393009','426177001','426783006','427084000','63593006',
                  '17338001','59931005','164917005','164934002','427172004','698252002']
    
    classes=sorted(classes)
    num_classes = len(classes)
    
    # Use your classifier here to obtain a label and score for each class. 
    feats_reshape,feats_external = get_features(data,header_data)
    feats_reshape = torch.tensor(feats_reshape,dtype=torch.float,device=device)
    feats_external = torch.tensor(feats_external,dtype=torch.float,device=device)
    model.eval()
    
    
    pred = model.forward(feats_reshape,feats_external)
    pred = torch.sigmoid(pred)
    
    current_score = pred.squeeze().cpu().detach().numpy()        
    current_label=np.where(current_score>0.15,1,0)       
    num_positive_classes = np.sum(current_label)
    
    normal_index=classes.index('426783006')
    max_index=np.argmax(current_score)     
           
    ##至少有一个标签，如果所有标签都没有，就将概率最大的设为1       
    if num_positive_classes==0:
        current_label[max_index]=1  
        
    return current_label, current_score, classes