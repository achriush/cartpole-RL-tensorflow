""" Trains an agent with (stochastic) Policy Gradients on CartPole-v1. Uses OpenAI Gym. """
import os
import sys
import gym
import numpy as np
import cPickle as pickle

from Qlearning_tf import neuron_network_for_Q_learning as DQN

from gym import wrappers

def sqrt ( n):
  eps= 0.001
  l=0.0
  r=float(n)
  while(r-l>eps):
    mid = (l+r )/2
    if mid*mid >n :
      r=mid
    else:
      l=mid
  return l
# Interested Game
game = 'CartPole-v0'

env = gym.make(game)

# env = wrappers.Monitor(env, './Records/{}'.format(game),force=True)

input_num=env.observation_space.shape[0]

# hyperparameters

PM=DQN(_D=input_num+2,_H=256,_W=256,_learning_rate=0.00618,_batch_size=10,_learning_rate_batch=10000000)




render = True # watch the game?

# model initialization

observation = env.reset()

running_reward = None

reward_sum = 0
episode_number = 0


epslon = 0.1



while episode_number < 1000000:
  PM.alpha = -0.1 / sqrt(episode_number +1)
  if render and reward_sum >200:
  	env.render()
  
  x = observation #x = prepro(observation)

  # forward the policy network and sample an action from the returned probability
  # aprob - action probability, h - hidden layer output

  QX0=PM.get_Q(x,[0,1])
  QX1=PM.get_Q(x,[1,0])

  action = 0 if QX0>QX1 else 1
  #print QX0,QX1,action
  action ^= 1 if np.random.uniform() < epslon else 0 # roll the dice!
  # grad from loss function to f_w that encourages the action to be y (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # this defines a direction, but how much is is encouraged is defined later by discouted return (MC. like method)

  observation, reward, done, info = env.step(action)

  next_Q = 0 if done else max (PM.get_Q(observation,[0,1]) , PM.get_Q(observation,[1,0]) ) 
  reward_sum += reward
  actions = []
  actions.append(action)
  actions.append(0 if action == 1 else 1)
  PM.add_info(x,actions,reward,next_Q,PM.get_Q(x,actions))



  if done: # an episode finished
    
    episode_number += 1


    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    if( episode_number % 200 == 0 ):
      print 'Episode No. %d:  reward total = %f. running mean: %f learing rate: %f' % (episode_number, reward_sum, running_reward , PM.learning_rate)
    reward_sum = 0
    observation = env.reset() # reset env


env.close()
# upload for evaluation
# gym.upload('./Records/{}'.format(game), api_key="sk_4IdEHBY4RN2i4UhDu99GMg")
