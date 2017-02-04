""" Trains an agent with (stochastic) Policy Gradients on CartPole-v1. Uses OpenAI Gym. """
import os
import sys
import gym
import numpy as np
import tensorflow as tf



class neuron_network_for_Q_learning(object):
  def sigmoid(cls,x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

  def __init__(cls,_D,_H,_W,_learning_rate,_batch_size,_learning_rate_batch):
    cls.D=_D
    cls.H=_H
    cls.W=_W
    
    cls.learning_rate=_learning_rate
    cls.learning_rate_batch = _learning_rate_batch
    cls.batch_size = _batch_size
    cls.count = 0
    
    
    cls.shapes = {
        'wd1' : [_D,_H] ,
        'wd2' : [_H,_W] ,
        'wd3' : [_W,1]
     
    }

    cls.weights = {
        'wd1' : tf.Variable(tf.random_normal(cls.shapes['wd1'], stddev=0.01) ),
        'wd2' : tf.Variable(tf.random_normal(cls.shapes['wd2'], stddev=0.01) ),
        'wd3' : tf.Variable(tf.random_normal(cls.shapes['wd3'], stddev=0.01) ),
          
    }
        
    cls.input = []
    cls.next_Qs = []
    cls.rewards = []
    cls.Q_values = []
    

    cls.gamme = 0.8
    cls.info_num = 0

    cls.print_flag = True
    
    cls.x = tf.placeholder(tf.float32, [None, cls.D])
    cls.y = tf.placeholder(tf.float32, [None, 1])
    cls.keep_prob = 1.
    cls.input_data = []
    
    cls.init_model()
    
    cls.sess=tf.Session()
    cls.sess.run(cls.init)

  def model_forward(cls,_X,_dropout):
        
    _X = tf.nn.dropout(_X, _dropout)
    d1 = tf.nn.relu(tf.matmul(_X , cls.weights['wd1']), name="d1")
    d2x = tf.nn.dropout(d1, _dropout)
    d2 = tf.nn.relu(tf.matmul(d2x, cls.weights['wd2']), name="d2")

    dout =tf.nn.dropout(d2,_dropout)
    # Output, class prediction
    out = tf.matmul(dout, cls.weights['wd3'])
    return out
  
  def init_model(cls):
    cls.pred = cls.model_forward(cls.x,cls.keep_prob)
    cls.cost = tf.pow(cls.pred - cls.y, 2)
    cls.optimizer = tf.train.AdamOptimizer(learning_rate=cls.learning_rate).minimize(cls.cost)
    cls.init = tf.initialize_all_variables()


  def policy_backward(cls,losses,w1_input,w2_input,w3_input):

    """ backward pass. (eph is array of intermediate hidden states) """

    dW3 = np.dot(w3_input.T,losses).ravel()
    #print losses
    dp = np.outer(losses, cls.model['w3'])

    dp[dp<0] = 0
    #print dp.shape
    dW2 = np.dot(w2_input.T, dp).ravel()

    dh = np.dot(dp,cls.model['w2'])

    dh[dh<0] = 0
    #print dh.shape
    dW1 = np.dot(w1_input.T , dh).ravel()

    cls.count+=1
    dW1=dW1.reshape(cls.H,cls.D)
    dW2=dW2.reshape(cls.W,cls.H)
    dW3=dW3.reshape(cls.W)
    cls.model['w1']+=cls.learning_rate*dW1
    cls.model['w2']+=cls.learning_rate*dW2
    cls.model['w3']+=cls.learning_rate*dW3

  def get_Q(cls,_X,action):
    input_x=np.append(_X,action).reshape(-1,6)
    #print cls.sess.run(cls.pred ,feed_dict={cls.x:input_x })
    ret = cls.sess.run(cls.pred ,feed_dict={cls.x:input_x })
    return ret

  def add_info(cls,x,action,reward,next_Q,Q_value):

    cls.input_data.append(np.append(x,action))

    cls.next_Qs.append(next_Q)
    cls.rewards.append(reward)

    cls.info_num+=1

    if cls.info_num % cls.batch_size == 0:
      cls.update_model()

  def update_model(cls):
    start_pos = max( 0 , cls.info_num - cls.batch_size )
    
    list_input_data = cls.input_data [start_pos : ]
    rewards         = cls.rewards    [start_pos : ]
    next_Qs         = cls.next_Qs    [start_pos : ]
    
    input_data = np.vstack(list_input_data)
    target_value = np.vstack(rewards) + (np.vstack(next_Qs)*cls.gamme)
    cls.sess.run(cls.optimizer , feed_dict = { cls.x : input_data , cls.y : target_value })
    
    

    if (cls.count % cls.learning_rate_batch ==0 ) :
      cls.learning_rate *= 0.9999

    





