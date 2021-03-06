from networks.Control_Q_network import ControlDQNetwork
from networks.Atari_Q_Network import AtariDQNetwork
from replay_buffers.Replay_Buffer import ReplayBuffer
import tensorflow as tf
import math
import random
import numpy as np
import abc


class DQNBase:
    def __init__(self,
                 action_size,
                 environment_type,
                 replay_buffer_size,
                 min_replay_buffer_size,
                 mini_batch_size,
                 gama,
                 learning_rate,
                 epsilon_min,
                 update_target_network_after):
        """
        :param action_size: shows the number of possible action for the environment e.g. 5 in case og Breakout game
        :param environment_type: either control (e.g. CartPole, mountain car, ... ) or atari (e.g. breakout, pong, ...)
        :param replay_buffer_size:capacity of experience replay memory
        :param mini_batch_size:size of the mini_batch
        :param gama: value of gamma discount factor
        :param learning_rate: value of optimizer learning rate
        :param epsilon_min epsilon is equal to one at the beginning and will anneal to epsilon_min gradually
        :param update_target_network_after: target network will update after each update_target_network_after actions
        """
        self.action_size = action_size
        self.environment_type = environment_type
        self.replay_buffer_size = replay_buffer_size
        self.min_replay_buffer_size = min_replay_buffer_size
        self.mini_batch_size = mini_batch_size
        self.gama = gama
        self.learning_rate = learning_rate
        self.epsilon_min = epsilon_min
        self.update_target_network_after = update_target_network_after

        self.epsilon = 1
        self.epsilon_decay = 1000  # after epsilon_decay*2 steps epsilon reaches to epsilon_min

        self.replay_memory = ReplayBuffer(self.replay_buffer_size)

        self.model = None
        self.target_model = None
        self.create_network()
        self.update_target_network()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.Huber(), metrics=['accuracy'])

        self.total_steps_counter = 0
        self.update_target_counter = 0

    def create_network(self):
        """
        Creates a DQNetwork based on environment type
        :return: None
        """
        if self.environment_type == "control":
            self.model = ControlDQNetwork(self.action_size)
            self.target_model = ControlDQNetwork(self.action_size)
        elif self.environment_type == "atari":
            self.model = AtariDQNetwork(self.action_size)
            self.target_model = AtariDQNetwork(self.action_size)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        self.update_target_counter = 0
        print("** Target model updated **")

    def act(self, state, mode):
        if mode == "train":
            self.total_steps_counter += 1
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(
                -1.0 * self.total_steps_counter / self.epsilon_decay)

            if random.random() < self.epsilon:
                return random.sample(range(self.action_size), 1)[0]
            else:
                return np.argmax(self.model.predict(state))
        elif mode == "test":
            return np.argmax(self.model.predict(state))

    def save_model(self, env_name, episode_num):
        self.model.save("saved_model/nature_dqn_" + str(env_name) + '_' + str(episode_num))

    def load_model(self, env_name, episode_num):
        tf.print("Loading Model ...")
        self.model = tf.keras.models.load_model("saved_model/nature_dqn_" + str(env_name) + '_' + str(episode_num))

    @abc.abstractmethod
    def train(self, terminal):
        raise NotImplementedError

    @abc.abstractmethod
    def replay_buffer_append(self, transition):
        raise NotImplementedError
