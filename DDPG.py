# We will use actor critic based Deep Q-learning Learning algorithm
# Will use double Q learning ie generalized network and a target network.
# Will use at least two class, one for Q-Learning and other for replay buffer.
# Critic network depends on both s and a, where as actor network depends only on s.
# Target network will use update rule t = tau*param + (1-tau)*param.
# Will use batch norm for training stability.
# Will create exploration policy by adding a noise process to actor policy.
# For critic we will use L2 weight decay and discount factor of 0.99.
# Used tau = 0.99.
# Learning rate for actor and critic are 10^-4 and 10^-3.
# Neural netowrks use ReLU activation function for all the hidden layer.
# The output layer of the actor is tanh layer to bound the action.
# For exploration noise process, O-U process is used with theta=0.15 and sigma=0.2.
# The target actor is just generalized actor plus O-U noise -> will need a class for noise.
# Will need to bound the actions according to the env action space.
import os
import tensorflow.compat.v1 as tf
import numpy as np
import random
import gym
from tensorflow.compat.v1.initializers import random_uniform

tf.disable_eager_execution()

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0 =None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, maxlen, input_size, n_actions):
        self.maxlen = maxlen
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.maxlen, input_size))
        self.new_state_memory = np.zeros((self.maxlen, input_size))
        self.action_memory = np.zeros((self.maxlen, n_actions))
        self.reward_memory = np.zeros(self.maxlen)
        self.terminal_memory = np.zeros(self.maxlen)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.maxlen
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        min_index = min(self.maxlen, self.mem_cntr)
        batch = np.random.choice(min_index, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        next_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]


        return states, actions, next_states, rewards, dones



class actor(object):
    def __init__(self, n_actions, input_dims, learning_rate, name, sess, fc1_dims, fc2_dims, action_bound, batch_size=64):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.name = name
        self.sess = sess
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.build_network()
        self.params = tf.trainable_variables(scope = self.name)

        #self.saver = tf.train.Saver()
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params,
                                                         -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.params))


    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims], name = "inputs")

            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions])

            f1 = 1/np.sqrt(self.fc1_dims)

            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, activation=tf.nn.relu,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))

            batch1 = tf.layers.batch_normalization(dense1)

            f2 = 1/np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(batch1, units=self.fc2_dims, activation=tf.nn.relu,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))

            batch2 = tf.layers.batch_normalization(dense2)

            f3 = 0.003

            mu = tf.layers.dense(batch2, units = self.n_actions,
                                 activation='tanh',
                                 kernel_initializer=random_uniform(-f3, f3),
                                 bias_initializer=random_uniform(-f3, f3))

            self.mu = tf.multiply(mu, self.action_bound)


    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})


    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                self.action_gradient: gradients})


class critic(object):
    def __init__(self, n_actions, input_dims, learning_rate, name, sess, fc1_dims, fc2_dims, batch_size=64):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.name = name
        self.sess = sess
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)


    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, 1],
                                           name='targets')

            f1 = 1 / np.sqrt(self.fc1_dims)

            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, activation=tf.nn.relu,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))

            batch1 = tf.layers.batch_normalization(dense1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(batch1, units=self.fc2_dims,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))

            batch2 = tf.layers.batch_normalization(dense2)

            action_in = tf.layers.dense(self.actions, units = self.fc2_dims)

            state_actions_temp = tf.add(batch2, action_in)

            state_actions = tf.nn.relu(state_actions_temp)

            f3 = 0.003

            self.q = tf.layers.dense(state_actions, units = 1,
                                     kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3,f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize,
                             feed_dict = {self.input: inputs,
                                          self.actions: actions,
                                          self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict = {self.input: inputs,
                                          self.actions: actions})


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer_1_size=400,
                 layer_2_size=300, batch_size = 64):
        self.gamma = gamma
        self.tau=tau
        self.input_dims = input_dims
        self.memory = ReplayBuffer(max_size, *self.input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = actor(n_actions, self.input_dims, alpha, 'Actor',
                           self.sess, layer_1_size, layer_2_size, env.action_space.high)

        self.critic = critic(n_actions, self.input_dims, beta, 'Critic',
                             self.sess, layer_1_size, layer_2_size)

        self.target_actor = actor(n_actions, self.input_dims, alpha, 'TargetActor',
                           self.sess, layer_1_size, layer_2_size, env.action_space.high)

        self.target_critic = critic(n_actions, self.input_dims, beta, 'TargetCritic',
                             self.sess, layer_1_size, layer_2_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = \
            [self.target_critic.params[i].assign(
                tf.multiply(self.critic.params[i], self.tau)\
                + tf.multiply(self.target_critic.params[i], 1. - self.tau))
                for i in range(len(self.target_critic.params))]

        self.update_actor = \
            [self.target_actor.params[i].assign(
                tf.multiply(self.actor.params[i], self.tau) \
                + tf.multiply(self.target_actor.params[i], 1. - self.tau))
                for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first = False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau

        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, new_state, reward, done = self.memory.sample_buffer(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))

        target = []

        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target)

        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()


env = gym.make("Pendulum-v0")
agent = Agent(alpha=0.0001, beta=0.001, input_dims=[3], tau=0.001, env=env,
              batch_size=64, layer_1_size=400, layer_2_size=300, n_actions=1)
np.random.seed(0)
score_history = []
for i in range(1000):
    obs=env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)

        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)
    print("episode", i, 'score %.2f' % score)










        