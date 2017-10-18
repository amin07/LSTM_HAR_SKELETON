'''
Created on Oct 6, 2017

@author: Amin
'''

import os
###### data parameters #########
num_person = 20;
T  = 15;
considered_joints = [6,10,7,11,4,5,9,22,24];
# considered_joints = [6,10,22,24];
dir_name = 'C:/Users/Amin/Downloads/CPS_project/data_sets/nturgbd_skeletons/nturgb_d_skeletons'
file_names = (os.listdir(dir_name))
class_ids = [3,4,7,10,20,21,23,31,38]
# class_ids = [4,10,23,31,38]

class_temp_to_real = {f:r for f,r in enumerate(class_ids)}
class_real_to_temp = {r:f for f,r in enumerate(class_ids)}
class_names = ['A'+str(c).zfill(3) for c in class_ids]
replications = ['R001']
persons = ['P001', 'P002', 'P003','P004','P005', 'P006','P007', 'P008','P009','P010']
# persons = ['P001']
batch_size = 5
###### data parameters #########