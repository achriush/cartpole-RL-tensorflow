""" Trains an agent with (stochastic) Policy Gradients on CartPole-v1. Uses OpenAI Gym. """
import os
import sys
import gym
import numpy as np








class logistic_regression_policy_model(object):
  
  def sigmoid(cls,x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

  def __init__(cls,_D,_learning_rate,_batch_size):
    cls.D=_D
    
    cls.learning_rate=_learning_rate

    cls.W = np.random.randn(_D)/ np.sqrt(_D)

    cls.dW=np.zeros_like(cls.W)

    cls.batch_size=_batch_size

    cls.count=0

    cls.xs = []
    cls.actions = [] 
    cls.probs = []
    cls.rewards = []

    cls.tot_rewards=None


    cls.gamma = 0.9

  def policy_forward(cls,x):

    logp = np.dot(cls.W,x)
    p = cls.sigmoid(logp)
    return p # return probability of taking action 2, and hidden state

  def policy_backward(cls,epdlogp,epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW = np.dot(epx.T, epdlogp).ravel()
    cls.count+=1
    cls.dW+=dW
    if cls.count % cls.batch_size == 0:
      cls.W+=cls.learning_rate*cls.dW
      cls.dW=np.zeros_like(cls.dW)


  def get_action_prob(cls,_X):

    prob = cls.policy_forward(_X)

    return prob

  def add_info(cls,x,aprob,action,reward):
    cls.xs.append(x)
    cls.actions.append(action)
    cls.probs.append(aprob)
    cls.rewards.append(reward)


  def update_model(cls):
    epx = np.vstack(cls.xs)
    epdlogp = np.vstack(cls.actions) - np.vstack(cls.probs)
    epr = np.vstack(cls.rewards)


    cls.xs,cls.actions,cls.rewards,cls.probs = [],[],[],[] # reset array memory


    discounted_epr = cls.discount_rewards(epr)

    cls.tot_rewards = discounted_epr if cls.tot_rewards is None else np.append(cls.tot_rewards,discounted_epr)


    discounted_epr -= np.mean(cls.tot_rewards)
    
    discounted_epr /= np.std(discounted_epr)
    oldepd=np.copy(epdlogp)
    epdlogp *= discounted_epr 


    cls.policy_backward(epdlogp,epx)

  

  def discount_rewards(cls,r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
      if r[t] < 0: running_add = 0 # reset the sum, game boundary
      running_add = running_add * cls.gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r



