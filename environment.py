import pandas as pd
import numpy as np
import random
class Boscun:
    """Environment API to navigate environment through states, actions, and rewards
    """
    def __init__(self):
        self.s = None # state
        self.a = None # action
        self.r = None # reward
        self.cum_r = 0 # Cumulative Reward
        self.episode_states = None # Episode (trip ID)
        self.episode_rewards = None
        self.episode_step = 0
        # Training & Test Data
        self.train_data = pd.read_csv('output/states_train.csv')
        self.train_rewards = pd.read_csv('output/rewards_train.csv')
        self.test_data = pd.read_csv('output/states_test.csv')
        self.test_rewards = pd.read_csv('output/rewards_test.csv')
        self.num_states = len(self.train_data.columns) - 2
        self.num_actions = 2 # BUY or WAIT
        self.episode_id = None
        self.episode_ids = self.train_rewards.trip.unique()
        self.test_episode_ids = self.test_rewards.trip.unique()
        self.test_index = 0
        
    def reset(self):
        """Start a new episode, selected randomly
        """
        self.episode_id = random.choice(self.episode_ids)
        return self.set_episode(self.episode_id)

    def test_reset(self):
        """Start a next episode, tracked to run each test episode once
        """
        if self.test_index >= len(self.test_episode_ids)-1:
          self.test_index = 0
          return None, True
        self.episode_id = self.test_episode_ids[self.test_index]
        next_s = self.set_episode(self.episode_id, train = False)
        self.test_index += 1
        return next_s, False
          
        
    def set_episode(self, ep_id, train = True):
        """Subset states and rewards by current episode for reward & state lookups
        """
        if train:
          tmp = self.train_data
          tmp_r = self.train_rewards
        else:
          tmp = self.test_data
          tmp_r = self.test_rewards
        self.episode_states = tmp[tmp.trip == ep_id]
        self.episode_rewards = tmp_r[tmp_r.trip == ep_id]
        self.episode_step = 0
        self.s = self.episode_states.iloc[0,2:]
        return self.s
        
    def step(self, a):
        """Given a current state and action, calculate the reward for action and next state
        """
        self.a = a
        wait_return = self.episode_rewards.iloc[self.episode_step].daily_return
        self.r = a * wait_return # 0 if BUY, daily return if WAIT
        if self.r > 0: self.r *= 0.25
        self.cum_r += self.r
        self.s = self.episode_states.iloc[self.episode_step + 1, 2:]
        terminal = self.episode_rewards.iloc[self.episode_step + 1].terminal
        done = terminal or a == 0 # State is terminal if ticket is bought
        self.episode_step += 1
        
        return self.s, self.r, done
