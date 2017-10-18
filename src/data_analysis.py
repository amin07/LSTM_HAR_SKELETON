'''
Created on Sep 29, 2017

@author: Amin
'''
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from shutil import copyfile
import matlab.engine
import re
import time
import numpy as np
import tensorflow as tf
import sys
import models
from network_params import *
from sklearn.model_selection import train_test_split
from PIL.BmpImagePlugin import SAVE
import matplotlib as mpl



#######data splitting#######
X_dat = np.load('data/X_dat_large.npy')    # 305X4X10X3
y_dat = np.load('data/y_dat_large.npy').astype(np.int32)
y_dat = np.array([class_real_to_temp[f] for f in y_dat])
# file_names = np.load('data/dat_filename.npy')
X_train, X_test, y_train, y_test = train_test_split(X_dat, y_dat, test_size=0.30, random_state=43)

# print (np.shape(X_dat))
# X_train = X_dat[:40]
# y_train = y_dat[:40]
# X_test = X_dat[40:]
# y_test = y_dat[40:]


# print (X_dat[:10], y_dat[:10])
# sys.exit()
#######data splitting#######

def gen_batches(num_steps=10, batch_size=33):

#     print ('x y shapes', np.shape(X_dat), np.shape(y_dat))
    data_len = np.shape(X_train)[0]
    batch_count = data_len//batch_size
    for i in range(batch_count):
        x = X_train[i*batch_size:(i+1)*batch_size]
        y = y_train[i*batch_size:(i+1)*batch_size]
        yield (x,y)
 

def gen_epochs(num_epochs=3, num_steps=10, batch_size=33):
    for i in range(num_epochs):
            yield gen_batches(num_steps=num_steps, batch_size=batch_size);


def train_network(g, num_epochs, num_steps = 10, batch_size = 33, verbose = True, save=False):
    tf.set_random_seed(2345)
    saved_data = ''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        
        
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            batch_cnt = 0
            for (X, Y) in epoch:
#                 Y = np.array([[i]*num_steps for i in Y])
                batch_cnt += 1
                training_state = None
                
                feed_dict={g['x']: X, g['y']: Y, g['b_size']:np.shape(X)[0]}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, _, acc, preds, pred2, fin_layer_op = sess.run([g['total_loss'],
                                                      g['train_step'], g['acc'], g['preds'], g['pred2'], g['final_layer_op_']],
                                                             feed_dict)
                training_loss += training_loss_
                training_losses.append(training_loss)
            
            print ('avg training loss after epoch, ', idx,': ',training_loss/batch_cnt)
#             print ('final layerop/mem', fin_layer_op)    
#         x_data = X_train[:batch_size]
#         y_data = np.array([[i]*num_steps for i in y_train[:batch_size]])
        
        x_data = X
        y_data = Y
        feed_dict = {g['x']:x_data, g['y']:y_data,g['b_size']:np.shape(x_data)[0]}
#         print ('feed_dict', feed_dict)
        preds,pred2,acc = sess.run([g['preds'],g['pred2'],g['acc']], feed_dict)
        
#         print ('total loss after training on data 1', total_loss)
        print (preds)
#         print (y_data)
        print (pred2)
        print (y_data)
        print ('accuracy', acc)
#                         
        if isinstance(save, str):
            g['saver'].save(sess, save)
    
    return training_losses

def get_accuracy(g, checkpoint):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)
        
        for idx, epoch in enumerate(gen_epochs(5, T, batch_size)):
            training_loss = 0
            batch_cnt = 0
            for (X, Y) in epoch:
#                 Y = np.array([[i]*num_steps for i in Y])
                batch_cnt += 1
                x_data = X
                y_data = Y
                feed_dict = {g['x']:x_data, g['y']:y_data,g['b_size']:np.shape(x_data)[0]}
#                 print ('feed_dict', feed_dict)
                preds, pred2, acc = sess.run([g['preds'],g['pred2'], g['acc']], feed_dict)
                
                print ('batch no. ', batch_cnt)
                print ('actual label, ', y_data)
                print ('predicted label ,', pred2)
                print ('accuracy ',acc)
                print ('##################')
                
#         x_data = X_train[:batch_size]
#         y_data = np.array([[i]*num_steps for i in y_train[:batch_size]])
            break
#         x_data = X_train[batch_size:2*batch_size]
#         y_data = y_train[batch_size:2*batch_size]
        
        for i in range(len(X_test)//batch_size):
            print ('##################test data')
            x_data = X_test[batch_size*i:(i+1)*batch_size]
            y_data = y_test[batch_size*i:(i+1)*batch_size]
            feed_dict = {g['x']:x_data, g['y']:y_data,g['b_size']:np.shape(x_data)[0]}
    #         print ('feed_dict', feed_dict)
            preds, pred2, acc = sess.run([g['preds'],g['pred2'], g['acc']], feed_dict)
            
    #         print ('total loss after training on data 1', total_loss)
            print ((preds))
            print (y_data)
            print (pred2)
            print ('accuracy ',acc)
        
def get_data_details():
    
    fixed_part = ['C003'+p+r+c for c in class_names for r in replications for p in persons]
    
    f_names = []
    for f in file_names:
        for fx in fixed_part:
            if fx in f:
                f_names.append(f) 
                break
    
    print ('total sequences', len(f_names))
    sample_per_class = {}
    sample_per_actor = {}
    sample_per_season = {}
    for fname in f_names:
        try:
            sample_per_class[fname[-12:-9]] += 1
        except:
            sample_per_class[fname[-12:-9]] = 1
            
        try:
            sample_per_actor[fname[8:12]] += 1
        except:
            sample_per_actor[fname[8:12]] = 1
    
        try:
            sample_per_season[fname[0:4]] += 1
        except:
            sample_per_season[fname[0:4]] = 1
    print (sample_per_class)
    print (sample_per_actor)
    print (sample_per_season)
   
    return f_names

def visualize_data():
    print ('data visualization')

    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    clrs = ['r','g','b']
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    z = np.linspace(0, 16, 15)
#     print (z)
    for i in range(45):
        X_dattp = np.transpose(X_dat[i,-3])
#         z = X_dattp[2]
        idc = i%9
        if idc==0:
            ax.plot(X_dattp[0], X_dattp[1], z,color=clrs[0])
        if idc==1:
            ax.plot(X_dattp[0], X_dattp[1], z,color=clrs[1])
        if idc==2:
            ax.plot(X_dattp[0], X_dattp[1], z,color=clrs[2])
    
    
    ax.legend()
    
    plt.show()
    

def create_data():
    '''
    f_name = 'S001C003P006R001A001.skeleton'
    filename = dir_name + '/' + f_name
    eng = matlab.engine.start_matlab()
    sampled_video = eng.preprocess_sequence(filename, T, considered_joints);
    print (np.shape(sampled_video))
    '''
    f_names = get_data_details()
    video_data = []
    class_data = []
    cnt = 0
    for f_name in f_names:
        filename = dir_name + '/' + f_name;
        eng = matlab.engine.start_matlab();
        sampled_video = eng.preprocess_sequence(filename, T, considered_joints);
        print ('video data ', cnt+1, 'processed')
        cnt += 1
        print ('sampled_video', sampled_video)
        print ('sampled_video shape', np.shape(sampled_video))
        if(len(sampled_video) is not 0):
            video_data.append(sampled_video)
            class_data.append(filename[-12:-9])
    
#     np.save('data/X_dat_large.npy', np.array(video_data))
#     np.save('data/y_dat_large.npy', np.array(class_data))
#     np.save('data/dat_filename_large.npy', np.array(f_names))
    
if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#     print (class_real_to_temp)
#     create_data()
    
#     f_names = get_data_details()
#     print (np.transpose(f_names))
#     visualize_data()
    st = time.clock()
    g = models.build_rnn_graph_per_joint(num_classes=len(class_ids), batch_size=batch_size,num_steps=T)
#     g = models.build_rnn_graph_per_joint(num_classes=len(class_ids), batch_size=batch_size,num_steps=T)
#     g = models.build_basic_rnn_graph_with_list3(num_classes=len(class_ids), batch_size=batch_size,num_steps=T)
#     train_network(g, num_epochs=3000, num_steps=T, batch_size=batch_size, save="saves/epoch_3k_large_9jt")  
    get_accuracy(g, 'saves/epoch_3k_large_9jt')
    print ('time req:', time.clock()-st)
