#!/usr/bin/env python3

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *
import sys

env = gym.make('CartPole-v0')

RNG_SEED = 1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 0.0001
gamma = 0.99

w_init = xavier_initializer(uniform=False)
b_init = tf.constant_initializer(0.1)

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.float32, shape=(None, output_units), name='y')

out = fully_connected(inputs=x,
                      num_outputs=output_units,
                      activation_fn=tf.nn.softmax,
                      weights_initializer=w_init,
                      weights_regularizer=None,
                      biases_initializer=b_init,
                      scope='fc')

all_vars = tf.global_variables()

pi = tf.contrib.distributions.Bernoulli(p=out, name='pi')
pi_sample = pi.sample()
log_pi = pi.log_prob(y, name='log_pi')

Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY = 25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')


track_steps = []
track_returns = []

# For LaTeX plotting
w1_plot = ''
w2_plot = ''
w3_plot = ''
w4_plot = ''
w5_plot = ''
w6_plot = ''
w7_plot = ''
w8_plot = ''
returns_plot = ''
steps_plot = ''

for ep in range(2001):
    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        env.render()
        action = sess.run([pi_sample], feed_dict={x: [obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action[0])
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    index = ep % MEMORY

    print(np.array(ep_states))

    _ = sess.run([train_op], feed_dict={x: np.array(ep_states),
                                        y: np.array(ep_actions),
                                        Returns: returns})

    track_steps.append(t)
    track_steps = track_steps[-MEMORY:]
    mean_steps = np.mean(track_steps)

    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)

    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY, mean_return))
    print("Mean number of steps over the last {} episodes is {}".format(MEMORY, mean_steps))

    with tf.variable_scope('fc', reuse=True):
        weights = sess.run(tf.get_variable('weights'))
        print("Weights:")
        print(weights)

    if ep % 20 == 0:
        w1_plot += str((ep, weights[0, 0]))
        w2_plot += str((ep, weights[0, 1]))
        w3_plot += str((ep, weights[1, 0]))
        w4_plot += str((ep, weights[1, 1]))
        w5_plot += str((ep, weights[2, 0]))
        w6_plot += str((ep, weights[2, 1]))
        w7_plot += str((ep, weights[3, 0]))
        w8_plot += str((ep, weights[3, 1]))
        returns_plot += str((ep, mean_return))
        steps_plot += str((ep, mean_steps))

print('w1:', w1_plot)
print('w2:', w2_plot)
print('w3:', w3_plot)
print('w4:', w4_plot)
print('w5:', w5_plot)
print('w6:', w6_plot)
print('w7:', w7_plot)
print('w8:', w8_plot)
print('returns:', returns_plot)
print('steps:', steps_plot)

sess.close()
