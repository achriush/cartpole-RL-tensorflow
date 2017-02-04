""" Trains an agent with (stochastic) Policy Gradients on CartPole-v1. Uses OpenAI Gym. """
import os
import sys
import gym
import numpy as np



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

    cls.model = {}
    cls.model['w1'] = np.random.randn(_H,_D)/ np.sqrt(_D)
    cls.model['w2'] = np.random.randn(_W,_H)/ np.sqrt(_H)
    cls.model['w3'] = np.random.randn(_W)/ np.sqrt(_W)

    cls.input = {}
    cls.input['w1'] = []
    cls.input['w2'] = []
    cls.input['w3'] = []
    
    cls.next_Qs = []
    cls.rewards = []
    cls.Q_values = []
    
    cls.alpha = -0.1
    cls.gamme = 0.8
    cls.info_num = 0

    cls.print_flag = True

  def policy_forward(cls,w1_input):

    w2_input = np.dot( cls.model['w1'] , w1_input.T )

    w2_input[ w2_input < 0 ] = 0

    w3_input = np.dot(cls.model['w2'] , w2_input )

    w3_input[ w3_input < 0 ] = 0

    ret = np.dot(cls.model['w3'],w3_input)
    

    return w1_input,w2_input,w3_input,ret # return probability of taking action 2, and hidden state

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

    x,h,p,prob = cls.policy_forward(np.append(_X,action))

    return prob

  def add_info(cls,x,action,reward,next_Q,Q_value):

    w1_input,w2_input,w3_input,aprb=cls.policy_forward(np.append(x,action))

    cls.input['w1'].append(w1_input)
    cls.input['w2'].append(w2_input)
    cls.input['w3'].append(w3_input)


    cls.next_Qs.append(next_Q)
    cls.rewards.append(reward)
    cls.Q_values.append(Q_value)

    cls.info_num+=1

    if cls.info_num % cls.batch_size == 0:
      cls.update_model()

  def update_model(cls):
    start_pos = max( 0 , cls.info_num - cls.batch_size )
    w1_input=cls.input['w1'][ start_pos : ]
    w2_input=cls.input['w2'][ start_pos : ]
    w3_input=cls.input['w3'][ start_pos : ]

    w1_input = np.vstack(w1_input)
    w2_input = np.vstack(w2_input)
    w3_input = np.vstack(w3_input)

    rewards=cls.rewards[start_pos : ]
    next_Qs=cls.next_Qs[start_pos : ]
    Q_values=cls.Q_values[start_pos : ]

    delta_Q =  np.vstack(Q_values) - ( np.vstack(rewards) + (np.vstack(next_Qs)*cls.gamme) )

    delta_Q *= cls.alpha

    cls.policy_backward(delta_Q,w1_input,w2_input,w3_input)
    a,b,c,d =cls.policy_forward(w1_input)
    if cls.print_flag == False :
      print "----------new batch------------"
      print "Q before"
      print Q_values
      print "Q after"
      print d
      print "target Q"
      print  np.vstack(rewards) + (np.vstack(next_Qs)*cls.gamme) 
      print "----------next batch------------"
    #cls.print_flag = False
    if cls.info_num%10000 ==0 :
        print "Q function : %f ,next Q function : %f reward : %f " % (Q_values[0],next_Qs[0],rewards[0])
    

    if (cls.count % cls.learning_rate_batch ==0 ) :
        cls.learning_rate *= 0.9

    





