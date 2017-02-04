""" Trains an agent with (stochastic) Policy Gradients on CartPole-v1. Uses OpenAI Gym. """
import os
import sys
import gym
import numpy as np
import cPickle as pickle

from PolicyGridant import logistic_regression_policy_model as LR_model

from gym import wrappers

# Interested Game
game = 'CartPole-v0'

env = gym.make(game)

# env = wrappers.Monitor(env, './Records/{}'.format(game),force=True)

input_num=env.observation_space.shape[0]

# hyperparameters

PM1=LR_model(_D=input_num,_learning_rate=0.618,_batch_size=2)
PM2=LR_model(_D=input_num,_learning_rate=0.618,_batch_size=2)
PM3=LR_model(_D=input_num,_learning_rate=0.618,_batch_size=2)




render = True # watch the game?

# model initialization

observation = env.reset()

running_reward = None

reward_sum = 0
episode_number = 0






while episode_number < 4000:
  
  if render and reward_sum >200:
  	env.render()
  
  x = observation #x = prepro(observation)

  # forward the policy network and sample an action from the returned probability
  # aprob - action probability, h - hidden layer output

  aprob1 = PM1.get_action_prob(x)
  aprob2 = PM2.get_action_prob(x)
  aprob3 = PM3.get_action_prob(x)

  print aprob1,aprob2,aprob3

  action1 = 1 if np.random.uniform() < aprob1 else 0 # roll the dice!
  action2 = 1 if np.random.uniform() < aprob2 else 0 # roll the dice!
  action3 = 1 if np.random.uniform() < aprob3 else 0 # roll the dice!

  # grad from loss function to f_w that encourages the action to be y (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # this defines a direction, but how much is is encouraged is defined later by discouted return (MC. like method)
  action = 0 if action1 + action2 +action3 <2 else 1
  observation, reward, done, info = env.step(action)

  #print type(observation)
  reward_sum += reward
  print np.append(x,action)
  
  PM1.add_info(x,aprob1,action,reward)
  PM2.add_info(x,aprob2,action,reward)
  PM3.add_info(x,aprob3,action,reward)



  if done: # an episode finished
    
    episode_number += 1

    PM1.update_model()
    PM2.update_model()
    PM3.update_model()


    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'Episode No. %d:  reward total = %f. running mean: %f' % (episode_number, reward_sum, running_reward)
    reward_sum = 0
    observation = env.reset() # reset env


env.close()
# upload for evaluation
# gym.upload('./Records/{}'.format(game), api_key="sk_4IdEHBY4RN2i4UhDu99GMg")
