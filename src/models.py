'''
Created on Oct 6, 2017
This file contains sequence models for training on skeleton data
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
from network_params import *
from tensorflow.python.ops.init_ops import TruncatedNormal
from tensorflow.python.ops.rnn_cell_impl import *
cnt = 0

class JointLSTM():
    '''
    custom lstm cells comprised of num_joints seperate lstm cells
    '''
    def __init__(self, num_joints=4, num_units_each_cell=5):        
        self.cells = [tf.nn.rnn_cell.LSTMCell(num_units_each_cell, state_is_tuple=True, initializer=TruncatedNormal()) for i in range(num_joints)]
        self.num_cells = num_joints
        self.mem_size_each = num_units_each_cell
        self.mem_size_global = num_units_each_cell * num_joints
        self.cell_op_states = [None]*self.num_cells
        
    def __call__(self, global_mem, inputs):
#         inputs = tf.unstack(inputs, axis=0)
        for i in range(self.num_cells):
            if self.cell_op_states[i]==None:
                init_state = self.cells[i].zero_state(batch_size, tf.float32)
#                 print ('init_state shape', init_state)
#                 print ('inputs shape', inputs[0])
                # without different scope, scope name's conflict
                self.cell_op_states[i] = self.cells[i](inputs[i],init_state, scope='cell_'+str(i))[0]  # taking only c state (m/hidden/op state has no use)
#                 print ('cell_op_shape', self.cell_op_states[i])
                
            else:
                self.cell_op_states[i] = \
                    self.cells[i](inputs[i], (self.cell_op_states[i], global_mem[i*self.mem_size_each:(i+1)*self.mem_size_each]), scope='cell_'+str(i))[0]
#             with tf.Session() as sess:
#                 print ('return shape of each LSTM ', sess.run([tf.shape(cell_op_states[0])]))
#             sys.exit()
        with tf.variable_scope('whole_op_gate'):
            W = tf.get_variable('W', [self.mem_size_global + self.num_cells * 3, self.mem_size_global])
            b = tf.get_variable('b', [self.mem_size_global], initializer=tf.constant_initializer(0.0))
    
        input_to_op_gate = tf.concat([global_mem, tf.reshape(inputs, [batch_size, 3 * self.num_cells])], axis=1)
        output_from_op_gate = tf.nn.sigmoid(tf.matmul(input_to_op_gate, W) + b)
        concatenated_mem = tf.nn.tanh(tf.concat([self.cell_op_states[i] for i in range(self.num_cells)], axis=1))
        final_mem = tf.multiply(output_from_op_gate, concatenated_mem) 
        return final_mem
            
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
 
 
def build_rnn_graph_per_joint(
    state_size=10,
    num_classes=10,
    batch_size=33,
    num_steps=15,
    learning_rate=1e-4):
    
     
    '''
    considering only one joints using dynamic rnn and lstm cells
    '''
    reset_graph()
     
    num_joints = len(considered_joints)
    b_size = tf.placeholder(tf.int32, [], name='batch_size_placeholder')
#      with tf.Session() as sess:
#        (sess.run([b_size], {b_size : 30}))
    x = tf.placeholder(tf.float32, [batch_size, num_joints, num_steps, 3], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size], name='labels_placeholder')
    
    rnn_inputs = tf.transpose(x, [2,1,0,3])
    
#     with tf.Session() as sess:
#         print('rnn_inputs shape', sess.run([tf.shape(rnn_inputs)]))
#     sys.exit()
    

#     num_joints = 9
    num_units_each_cell = 10
    cell = JointLSTM(num_joints=num_joints, num_units_each_cell=num_units_each_cell)
    final_mems = \
            tf.scan(lambda a, x: cell(a, x), rnn_inputs, initializer = tf.ones([batch_size, num_joints*num_units_each_cell]))
    
    print ('here')
#     with tf.Session() as sess:
#         print('final mems size', sess.run([tf.shape(final_mems)]))
#     sys.exit()
    final_layer_op = tf.squeeze(final_mems[-1,:,:])
    
    
    
#     with tf.Session() as sess:
#         print('final layer mem', sess.run([final_layer_op]))
#     sys.exit()
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [num_joints*num_units_each_cell, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
      
    final_logits = tf.matmul(final_layer_op, W) + b
     
#     with tf.Session() as sess:
#         print('logits shape', sess.run(tf.shape(final_logits)))
# #        print('final logits shape', sess.run(tf.shape(final_logits)))
# #         print ('labels shape', sess.run(tf.shape(tf.one_hot(y, num_classes))))
#     sys.exit()
     
     
    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_logits, labels=tf.one_hot(y, num_classes)))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
     
#      predictions = tf.nn.softmax(tf.squeeze(logits[:,-1,:]))  # taking final steps logits
    predictions = tf.nn.softmax(final_logits)
    pred2 = tf.argmax(predictions, 1)
#      with tf.Session() as sess:
#        print ('pred2 shape', sess.run(tf.shape(pred2)))
#        print ('prediction shape', sess.run(tf.shape(predictions)))
#      sys.exit()
#      correct_pred = tf.equal(pred2, tf.cast(tf.squeeze(y[:,-1]), tf.int64))
    correct_pred = tf.equal(pred2, tf.cast(y, tf.int64))
    accuracy_ = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
     
    return dict(
      x=x,
      y=y,
      total_loss=total_loss,
      train_step=train_step,
      preds=predictions,
      saver=tf.train.Saver(),
      b_size=b_size,  # for zero_state in case of dynamic batch_size
      acc=accuracy_,
      pred2=pred2,
      final_layer_op_ = final_layer_op
    )

def build_basic_rnn_graph_with_list3(
    state_size=10,
    num_classes=10,
    batch_size=33,
    num_steps=15,
    learning_rate=1e-4):     
    '''
    considering only one joints using dynamic rnn and lstm cells
    '''
    reset_graph()
    num_joints = len(considered_joints)
    b_size = tf.placeholder(tf.int32, [], name='batch_size_placeholder')
#      with tf.Session() as sess:
#        (sess.run([b_size], {b_size : 30}))
    x = tf.placeholder(tf.float32, [batch_size, num_joints, num_steps, 3], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size], name='labels_placeholder')
    # taking fourth joint
    rnn_inputs = tf.squeeze(x[:, -2, :])
#      rnn_inputs = tf.unstack(tf.reshape(tf.unstack(tf.transpose(x, perm=[0,2,1,3]), num_steps, axis=1),[-1, num_steps, num_joints*3]), num_steps, axis=1)
#      y_one_hot = tf.one_hot(y, depth=num_classes)
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(x), tf.shape(rnn_inputs)]))
#      sys.exit()
     
     
     
    
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(rnn_inputs)]))
#      sys.exit()    
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True, initializer=TruncatedNormal())
    init_state = cell.zero_state(b_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
 
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(rnn_outputs)]))
#      sys.exit()
 
    rnn_op_final = tf.squeeze(rnn_outputs[:, -1, :])
    with tf.variable_scope('softmax'):
      W = tf.get_variable('W', [state_size, num_classes])
      b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
     
    # logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    # This was as list, need to change so that as tensor
#      with tf.Session() as sess:
#        print('rnn_output shape', sess.run(tf.shape(rnn_outputs)))
#        print('final_state shape', sess.run(tf.shape(final_state)))
#      sys.exit()
#      logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs,[-1, state_size]), W) + b, [-1, num_steps, num_classes])
    # taking only final_logits (ta time num_steps) 
    final_logits = tf.matmul(rnn_op_final, W) + b
     
#      with tf.Session() as sess:
#         print('logits shape', sess.run(tf.shape(logits)))
# #        print('final logits shape', sess.run(tf.shape(final_logits)))
#         print ('labels shape', sess.run(tf.shape(tf.one_hot(y, num_classes))))
#      sys.exit()
     
     
    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_logits, labels=tf.one_hot(y, num_classes)))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
     
#      predictions = tf.nn.softmax(tf.squeeze(logits[:,-1,:]))  # taking final steps logits
    predictions = tf.nn.softmax(final_logits)
    pred2 = tf.argmax(predictions, 1)
#      with tf.Session() as sess:
#        print ('pred2 shape', sess.run(tf.shape(pred2)))
#        print ('prediction shape', sess.run(tf.shape(predictions)))
#      sys.exit()
#      correct_pred = tf.equal(pred2, tf.cast(tf.squeeze(y[:,-1]), tf.int64))
    correct_pred = tf.equal(pred2, tf.cast(y, tf.int64))
    accuracy_ = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
     
    return dict(
      x=x,
      y=y,
      init_state=init_state,
      final_state=final_state,
      total_loss=total_loss,
      train_step=train_step,
      preds=predictions,
      saver=tf.train.Saver(),
      b_size=b_size,  # for zero_state in case of dynamic batch_size
      acc=accuracy_,
      pred2=pred2
    )
def build_basic_rnn_graph_with_list2(
    state_size=10,
    num_classes=10,
    batch_size=33,
    num_steps=15,
    learning_rate=1e-4):
     
    '''
    considering only one joints
    '''
    reset_graph()
     
    num_joints = len(considered_joints)
    b_size = tf.placeholder(tf.int32, [], name='batch_size_placeholder')
#      with tf.Session() as sess:
#        (sess.run([b_size], {b_size : 30}))
    x = tf.placeholder(tf.float32, [batch_size, num_joints, num_steps, 3], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size], name='labels_placeholder')
     
    # unstack for making it a list
    rnn_inputs = tf.unstack(tf.squeeze(x[:, -2, :]), axis=1)
#      rnn_inputs = tf.unstack(tf.reshape(tf.unstack(tf.transpose(x, perm=[0,2,1,3]), num_steps, axis=1),[-1, num_steps, num_joints*3]), num_steps, axis=1)
#      y_one_hot = tf.one_hot(y, depth=num_classes)
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(x), tf.shape(rnn_inputs)]))
#      sys.exit()
     
     
     
    
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(rnn_inputs)]))
#      sys.exit()    
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(b_size, tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, dtype=tf.float32)
     
    rnn_op_last = tf.squeeze(rnn_outputs[-1])
 
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(rnn_op_last), tf.shape(final_state)]))
#      sys.exit()
 
    with tf.variable_scope('softmax'):
      W = tf.get_variable('W', [state_size, num_classes])
      b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
     
    # logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    # This was as list, need to change so that as tensor
#      with tf.Session() as sess:
#        print('rnn_output shape', sess.run(tf.shape(rnn_outputs)))
#        print('final_state shape', sess.run(tf.shape(final_state)))
#      sys.exit()
    logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b, [-1, num_steps, num_classes])
    # taking only final_logits (ta time num_steps) 
#      final_logits = tf.squeeze(tf.split(logits, num_or_size_splits=num_steps, axis=1)[-1])
    final_logits = tf.matmul(rnn_op_last, W) + b
#      with tf.Session() as sess:
#         print('logits shape', sess.run(tf.shape(logits)))
#         print('final logits shape', sess.run(tf.shape(final_logits)))
#         print ('labels shape', sess.run(tf.shape(tf.one_hot(y, num_classes))))
#      sys.exit()
     
     
    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_logits, labels=tf.one_hot(y, num_classes)))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
     
#      predictions = tf.nn.softmax(tf.squeeze(logits[:,-1,:]))  # taking final steps logits
    predictions = tf.nn.softmax(final_logits)
    pred2 = tf.argmax(predictions, 1)
#      with tf.Session() as sess:
#        print ('pred2 shape', sess.run(tf.shape(pred2)))
#        print ('prediction shape', sess.run(tf.shape(predictions)))
#      sys.exit()
#      correct_pred = tf.equal(pred2, tf.cast(tf.squeeze(y[:,-1]), tf.int64))
    correct_pred = tf.equal(pred2, tf.cast(y, tf.int64))
    accuracy_ = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
     
    return dict(
      x=x,
      y=y,
      init_state=init_state,
      final_state=final_state,
      total_loss=total_loss,
      train_step=train_step,
      preds=predictions,
      saver=tf.train.Saver(),
      b_size=b_size,  # for zero_state in case of dynamic batch_size
      acc=accuracy_,
      pred2=pred2
    )
 
 
def build_basic_rnn_graph_with_list(
    state_size=30,
    num_classes=10,
    batch_size=33,
    num_steps=10,
    learning_rate=1e-4):
 
    reset_graph()
     
    num_joints = len(considered_joints)
    b_size = tf.placeholder(tf.int32, [], name='batch_size_placeholder')
#      with tf.Session() as sess:
#        (sess.run([b_size], {b_size : 30}))
    x = tf.placeholder(tf.float32, [batch_size, num_joints, num_steps, 3], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size], name='labels_placeholder')
     
    rnn_inputs = tf.unstack(tf.reshape(tf.unstack(tf.transpose(x, perm=[0, 2, 1, 3]), num_steps, axis=1), [-1, num_steps, num_joints * 3]), num_steps, axis=1)
#      y_one_hot = tf.one_hot(y, depth=num_classes)
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(rnn_inputs)]))
#      sys.exit()
     
     
     
#      
#      with tf.Session() as sess:
#        print (sess.run([tf.shape(rnn_inputs)]))
#      sys.exit()    
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(b_size, tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
 
    with tf.variable_scope('softmax'):
      W = tf.get_variable('W', [state_size, num_classes])
      b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
     
    # logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    # This was as list, need to change so that as tensor
#      with tf.Session() as sess:
#        print('rnn_output shape', sess.run(tf.shape(rnn_outputs)))
#        print('final_state shape', sess.run(tf.shape(final_state)))
#      sys.exit()
    logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b, [-1, num_steps, num_classes])
    # taking only final_logits (ta time num_steps) 
    final_logits = tf.squeeze(tf.split(logits, num_or_size_splits=num_steps, axis=1)[-1])
     
#      with tf.Session() as sess:
#        print('final logits shape', sess.run(tf.shape(final_logits)))
#        print ('labels shape', sess.run(tf.shape(y)))
    # sys.exit()
     
     
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    predictions = tf.nn.softmax(final_logits)
    pred2 = tf.argmax(predictions, 1)
    correct_pred = tf.equal(pred2, tf.cast(y, tf.int64))
    accuracy_ = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#      print ('predictions : ', predictions)
#      print ('labels : ', y)
     
    return dict(
      x=x,
      y=y,
      init_state=init_state,
      final_state=final_state,
      total_loss=total_loss,
      train_step=train_step,
      preds=predictions,
      saver=tf.train.Saver(),
      b_size=b_size,  # for zero_state in case of dynamic batch_size
      acc=accuracy_,
      pred2=pred2
    )
