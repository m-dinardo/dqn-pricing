import numpy as np
import tensorflow as tf
import random
from collections import deque
from environment import Boscun
import csv

class DDQN:
  def __init__(self, env, params):
    #tf.reset_default_graph()
    self.sess = None
    self.env = env # Environment
    self.set_params(params)
    
  def create_graph(self):
    # Reset graph at each call to train
    tf.reset_default_graph()
    #self.set_params(params)
    self.create_placeholders()
    self.gen_online_network()
    self.update_target_network()
    self.loss_estimate()
    self.init_optimizer()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    #self.sess.run(tf.initialize_all_variables())
    
  def train(self, save_results=False):
    """Train DDQN
    save_results  If true will save training and test results to csv, file name controlled by param
    """
    self.create_graph() # Reset graph at every training event
    #tf.reset_default_graph()
    #self.sess.run(tf.global_variables_initializer())
    
    episode_results = [] # To be saved as CSV results
    total_rewards = []
    experiences = deque(maxlen=self.replay_size) # Memory bank of length self.replay_size
    tot_steps = 0
    for i in range(self.nepisodes):
      total_r = 0
      episode_steps = 0
      done = False
      state = self.env.reset()
      trip = str(self.env.episode_id)
      while not done and episode_steps < self.nsteps:
        s = state.reshape((1, self.num_states))
        
        # Random selection or Q selection
        if np.random.random() < self.epsilon:
          a = self.random_action()
        else:
          a = self.forward_selection(s)
        
        next_state, r, done = self.env.step(a)
        # Should be 0 on valuation of subsequent states at terminal state
        term_state_multiplier = 0.0 if done else 1.0
        # Add to memory bank
        experiences.append((state, a, r, next_state, term_state_multiplier)) 
        
        # Update episode stats
        total_r += r
        episode_steps += 1
        tot_steps += 1
        
        # Copy online weights to target weights if time
        if tot_steps % self.steps_update == 0:
          _ = self.sess.run(self.update_target)
        # Experience replay if sufficient replays available
        if len(experiences) > self.batch_size:
          self.update_step(experiences)
          
        state = next_state
      total_rewards.append(total_r)
      print('100 Episode MA: ', str(np.mean(total_rewards[-100:])), '    Episode', i, 'Total Reward', total_r, '  Steps in Episode', episode_steps, ' Epsilon', self.epsilon)
      episode_results.append([str(trip), total_r, episode_steps, self.epsilon, self.alpha, self.gamma])
      self.decay_epsilon() # Decay epsilon according to param policy
      if i % 2 == 0:
        self.test_results(int(i / 2))
    # Return training results
    if save_results:
      self.write_results(episode_results, self.fname)
      self.test_results('final')
    tf.reset_default_graph() # To avoid overlapping sessions
    #return episode_results
    return "Done"
  
  def test_results(self, exp_num):
    """Test DDQN
    exp_num   File identifier, test #
    """
    print('TESTING RESULTS:    ')
    episode_results = [] # To be saved as CSV results
    tot_steps = 0
    test_finished = False
    i = 1
    while not test_finished: 
      total_r = 0
      episode_steps = 0
      done = False
      state, test_finished = self.env.test_reset()
      if test_finished: break
      trip = str(self.env.episode_id)
      while not done:
        s = state.reshape((1, self.num_states))
        a = self.forward_selection(s)
        next_state, r, done, = self.env.step(a)
        # Update episode stats
        total_r += r
        episode_steps += 1
        tot_steps += 1
        state = next_state
      print('TEST Episode', i, 'Total Reward', total_r, '  Steps in Episode', episode_steps)
      episode_results.append([trip, total_r, episode_steps, self.gamma])
      i += 1

    self.write_results(episode_results, self.fname + '_test' + str(exp_num))
    
  
  def decay_epsilon(self):
    # Ad-hoc values, TODO: make these values contorlled by parameters
    if self.e_decay_type == 'exponential':
      self.epsilon *= 0.9995
    elif self.e_decay_type == 'linear':
      self.epsilon -= 1.0 / 900 # Want to decay from 1.0 to 0 by 900 steps
    # elif constant don't decay
      
  def random_action(self):
    # Choose random action
    a = np.random.randint(self.num_actions)
    return(a)
    
  def forward_selection(self, s):
    # Select maximum arg from output vector of feed-forward value function
    a = np.argmax(self.sess.run(self.q_s_estimate, feed_dict = {self.s : s}))
    return(a)
    
  def update_step(self, experiences):
    # 1. Random sample experiences from replay
    batch = random.sample(experiences, self.batch_size)
    s_l = np.array([b[0] for b in batch])
    a_l = np.array([b[1] for b in batch])
    r_l = np.array([b[2] for b in batch])
    ns_l = np.array([b[3] for b in batch])
    term_l = np.array([b[4] for b in batch])
    # 2. Perform optimizer step on experience replays
    _ = self.sess.run(self.optimizer, feed_dict = {
          self.s : s_l,
          self.a : a_l,
          self.r : r_l,
          self.ns : ns_l,
          self.termstate : term_l
        })
    
  def set_params(self, params):
    # Set parameters for learning
    self.alpha = params['alpha']
    self.gamma = params['gamma']
    self.epsilon = params['e_start']
    self.e_decay_type = params['e_decay_type'] # linear, exponential, or constant
    self.num_nodes = params['num_nodes']
    self.nepisodes = params['nepisodes']
    self.nsteps = 10000 # Don't terminate early
    self.replay_size = params['replay_size']
    self.batch_size = params['batch_size']
    self.steps_update = params['steps_update'] # Steps to update target weights
    #self.env = gym.make(params['environment'])
    self.num_states = self.env.num_states
    self.num_actions = self.env.num_actions
    self.outdir = params['outdir']
    self.fname = params['fname']
    self.regularization = params['l2regul']
    
  def create_placeholders(self):
    """Placeholders"""
    self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.num_states]) # state
    self.a = tf.placeholder(dtype=tf.int32, shape=[None]) # action
    self.r = tf.placeholder(dtype=tf.float32, shape=[None]) # Reward
    self.ns = tf.placeholder(dtype=tf.float32, shape=[None, self.num_states]) # next state
    self.termstate = tf.placeholder(dtype=tf.float32, shape=[None]) # whether or not terminal state
    self.episodes = tf.Variable(0.0, trainable=False)
    #self.episodes_inc = self.episodes.assign_add(1)
    
  def write_results(self, results, file_name):
    fpath = self.outdir + '/' + file_name + '.csv'
    with open(fpath, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(results)
  
  def create_nnet(self, in_state, online, cp_weights):
    # 3 Layer networks. Relu activation in hidden layers, linear output activation
    h1 = tf.layers.dense(in_state, self.num_nodes, activation = tf.nn.relu, trainable = online, name='hl1', reuse = cp_weights)
    h2 = tf.layers.dense(h1, self.num_nodes, activation = tf.nn.relu, trainable = online, name = 'hl2', reuse = cp_weights)
    h3 = tf.layers.dense(h2, self.num_nodes, activation = tf.nn.relu, trainable = online, name = 'hl3', reuse = cp_weights)
    output = tf.layers.dense(h3, self.num_actions, activation = None, trainable = online, name = 'output', reuse = cp_weights)
    return output
  
  def gen_online_network(self):
    # Create double Q Networks, 1 online network for action selection
    # and 1 network for state/action valuation
    with tf.variable_scope('online_network'):
      self.q_s_estimate = self.create_nnet(self.s, online = True, cp_weights = False)
      self.q_ns_estimate = tf.stop_gradient(self.create_nnet(self.ns, online = False, cp_weights = True))
    with tf.variable_scope('target_network'):
      self.target_q_val_estimate = tf.stop_gradient(self.create_nnet(self.ns, online = False, cp_weights = False))
  
  def update_target_network(self):
    # Copy weights from online network to target network
    online_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'online_network')
    target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
    cp_weights_list = []
    for i, var in enumerate(target_weights): 
      cp_weights_list.append(var.assign(online_weights[i]))
    
    self.update_target = tf.group(*cp_weights_list)
    
  def loss_estimate(self):
    # if experience is terminal
    # y_i = r_i
    # if experience is non-terminal, perform action selection with online Q netowrk
    # and action-value estimation with the target network
    # y_i = r_i + gamma * Q_target(max_a(Q_online(ns, a)))
    # Select action with online network
    q_target_action_max = tf.stack((tf.range(self.batch_size), tf.cast(tf.argmax(self.q_ns_estimate, axis=1), tf.int32)), axis=1)
    # Estimate value of subsequent state (0 if terminal state)
    q_ns = self.termstate * self.gamma * tf.gather_nd(self.target_q_val_estimate, q_target_action_max)
    target_y = self.r + q_ns # yi total
    y_hat = tf.gather_nd(self.q_s_estimate, tf.stack((tf.range(self.batch_size), self.a), axis=1))
    # loss as mse
    loss_fun = tf.reduce_mean(tf.square(target_y - y_hat))
    # Add regulatization
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'online_network'):
      if not 'bias' in var.name:
        loss_fun = loss_fun + self.regularization * 0.5 * tf.nn.l2_loss(var) #Computes half the L2 norm of a tensor without the sqrt
    self.loss_fun = loss_fun
    
  def init_optimizer(self):
    self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss_fun)
    
if __name__ == '__main__':
  test_params = {
    'alpha': .0001,
    'gamma': .995,
    'e_start': 1,
    'e_decay_type': 'exponential',
    'num_nodes': 50,
    'nepisodes': 20000,
    'replay_size': 10000,
    'batch_size': 320,
    'steps_update': 100,
    'outdir': 'results',
    'fname': 'boscun_agent',
    'l2regul': 0.00001
  }
  env = Boscun()
  learner = DDQN(env, test_params)
  print(learner.train(True))
