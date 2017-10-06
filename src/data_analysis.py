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


###### data parameters #########
num_person = 20;
T  = 10;
considered_joints = [6,10,22,24];
dir_name = 'C:/Users/Amin/Downloads/CPS_project/data_sets/nturgbd_skeletons/nturgb_d_skeletons'
file_names = (os.listdir(dir_name))
class_ids = [3,4,7,10,20,21,23,31,38]

class_temp_to_real = {f:r for f,r in enumerate(class_ids)}
class_real_to_temp = {r:f for f,r in enumerate(class_ids)}
class_names = ['A'+str(c).zfill(3) for c in class_ids]
replications = ['R001','R002']
persons = ['P001', 'P002', 'P003','P004','P005', 'P006']
###### data parameters #########


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()



def gen_epochs(num_epochs=3, num_steps=10, batch_size=33):
    X_dat = np.load('X_dat.npy')
    y_dat = np.load('y_dat.npy').astype(np.int32)
#     print (y_dat)
#     print (class_real_to_temp)
#     print (class_temp_to_real)
    y_dat = np.array([class_real_to_temp[f] for f in y_dat])
#     print (y_dat)
    
    
    
    print ('x y shapes', np.shape(X_dat), np.shape(y_dat))
    data_len = np.shape(X_dat)[0]
    batch_count = data_len//batch_size
    print (data_len, batch_count)
 
    for i in range(num_epochs):
        for j in range(batch_count):
            x = X_dat[j*batch_size:(j+1)*batch_size]
            y = y_dat[j*batch_size:(j+1)*batch_size]
            yield (x,y)
#         print (' end of epoch ', i+1)

def train_network(g, num_epochs, num_steps = 10, batch_size = 33, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        training_loss = 0
        for idx, (X, Y) in enumerate(gen_epochs(num_epochs=num_epochs)):
            training_state = None
            feed_dict={g['x']: X, g['y']: Y}
            if training_state is not None:
                feed_dict[g['init_state']] = training_state
            training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                  g['final_state'],
                                                  g['train_step']],
                                                         feed_dict)
            training_loss += training_loss_
            if idx%3==2:
                print ('loss after epochs ', (idx//3 + 1), training_loss)
                training_loss = 0
            
        '''
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            epoch_cnt = 0
            for X, Y in epoch:
                steps += 1
                epoch_cnt += 1
                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)
        '''
        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses


def build_basic_rnn_graph_with_list(
    state_size = 30,
    num_classes = len(class_ids),
    batch_size = 33,
    num_steps = T,
    learning_rate = 1e-4):

    reset_graph()
    
    num_joints = len(considered_joints)
    x = tf.placeholder(tf.float32, [batch_size, num_joints, num_steps, 3], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size], name='labels_placeholder')

    rnn_inputs = tf.unstack(tf.reshape(tf.unstack(tf.transpose(x, perm=[0,2,1,3]), num_steps, axis=1),[batch_size, num_steps, num_joints*3]), num_steps, axis=1)
#     y_one_hot = tf.one_hot(y, depth=num_classes)
#     with tf.Session() as sess:
#         print (sess.run([tf.shape(rnn_inputs)]))
#     sys.exit()
    
    
    
    
    with tf.Session() as sess:
        print (sess.run([tf.shape(rnn_inputs)]))
#     sys.exit()    
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    #logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    #This was as list, need to change so that as tensor
    with tf.Session() as sess:
        print('rnn_output shape', sess.run(tf.shape(rnn_outputs)))
        print('final_state shape', sess.run(tf.shape(final_state)))
#     sys.exit()
    logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs,[-1, state_size]), W) + b, [batch_size, num_steps, num_classes])
    # taking only final_logits (ta time num_steps) 
    final_logits = tf.squeeze(tf.split(logits, num_or_size_splits=num_steps, axis=1)[-1])
    
    with tf.Session() as sess:
        print('final logits shape', sess.run(tf.shape(final_logits)))
        print ('labels shape', sess.run(tf.shape(y)))
    #sys.exit()
    
    
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )


def main():
    print ('inside main')
    
    dir_name = 'C:/Users/Amin/Downloads/CPS_project/data_sets/nturgbd_skeletons/nturgb_d_skeletons'
    file_1 = 'S001C001P002R001A001.skeleton'
    file_2 = 'S001C003P002R001A001.skeleton'

    file_name1 = dir_name + '/' + file_1;
    
    
#     copyfile(dir_name+'/'+file_1, 'hair_side.skeleton')
#     copyfile(dir_name+'/'+file_2, 'hair_front.skeleton')
    
    file_names = (os.listdir('C:/Users/Amin/Downloads/CPS_project/data_sets/nturgbd_skeletons/nturgb_d_skeletons'))
    print ('total files', len(file_names))
    class_one = []
    for file_name in file_names:
        if 'P002' in file_name:
            class_one.append(file_name)
    print ('total class of person 2', len(class_one))
    class_count = {};
    class_count['hello']=1;
    
    for filename in class_one:
        try:
            class_count[filename[16:20]] += 1;
        except:
            class_count[filename[16:20]] = 0;
    print (class_count)
    print (class_one[:10])


def create_data():
    
    
    fixed_part = ['C003'+p+r+c for c in class_names for r in replications for p in persons]
#     f_names = [f for f in file_names for fx in fixed_part if fx in f]
    
    f_names = []
    for f in file_names:
        for fx in fixed_part:
            if fx in f:
                f_names.append(f) 
                break
    
#     print (f_names[82])
    '''
    print ('total sequences', len(f_names))
    action_class = 'A003'
    print ('total action ',action_class,len([f for f in f_names if action_class in f]))
    person_class = 'P003'
    print ('total action of person ',person_class,len([f for f in f_names if person_class in f]))
    print (f_names)
    '''
    
    video_data = []
    class_data = []
    cnt = 0
    for f_name in f_names[:100]:
        filename = dir_name + '/' + f_name;
        eng = matlab.engine.start_matlab();
        sampled_video = eng.preprocess_sequence(filename, T, considered_joints);
        print ('video data ', cnt+1, 'processed')
        cnt += 1
        print ('sampled_video', sampled_video)
        if(len(sampled_video) is not 0):
            video_data.append(sampled_video)
            class_data.append(filename[-12:-9])
    
    np.save('X_dat', np.array(video_data))
    np.save('y_dat', np.array(class_data))
    
    
if __name__ == '__main__':
    #main()
#     create_data()
        
#     X_dat = np.load('X_dat.npy')
#     y_dat = np.load('y_dat.npy')
#     print (np.shape(X_dat))
#     print (np.shape(np.transpose(X_dat, [0,2,1,3])))
#     print (np.reshape(np.transpose(X_dat, [0,2,1,3]), [4,10,12]))
#     print (y_dat)
    g = build_basic_rnn_graph_with_list()
    train_network(g, num_epochs=2000, num_steps=T, batch_size=33)  
    
#     filename = dir_name + '/' + file_1;
#     eng = matlab.engine.start_matlab();
#     sampled_video = eng.preprocess_sequence(filename, T, considered_joints);
#     print (sampled_video)
    