from replay_buffers.Prioritized_Replay_Buffer import PrioritizedReplayBuffer
from networks.Control_Q_network import ControlDQNetwork
from networks.Atari_Q_Network import AtariDQNetwork
import tensorflow as tf
import math
import random
import numpy as np


class PrioritizedDoubleDQNAgent:
    def __init__(self,
                 action_size,
                 environment_type,
                 replay_buffer_size=1000000,
                 min_replay_buffer_size=500,
                 mini_batch_size=32,
                 gama=0.99,
                 learning_rate=0.0007,
                 epsilon_min=0.1,
                 update_target_network_after=2000):
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
        self.minimum_priority = 0.01
        self.epsilon = 1
        self.epsilon_decay = 1000  # after epsilon_decay*2 steps epsilon reaches to epsilon_min

        self.replay_memory = PrioritizedReplayBuffer(self.replay_buffer_size)

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

    def get_q_values(self, state):
        return self.model.predict(state)

    def act(self, state):
        self.total_steps_counter += 1
        self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(
            -1.0 * self.total_steps_counter / self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.sample(range(self.action_size), 1)[0]
        else:
            return np.argmax(self.get_q_values(state))

    def train(self, terminal):
        if self.replay_memory.get_size() < self.min_replay_buffer_size:
            return

        data_in_mini_batch = self.replay_memory.sample(self.mini_batch_size)
        current_states = np.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_q_values_list = self.model.predict(current_states)

        next_states = np.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_q_values_list = self.target_model.predict(next_states)
        # doubleDQN unlike dqn, needs to prediction of next states q values using model
        double_next_q_values_list = self.model.predict(next_states)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, next_state, done) in enumerate(data_in_mini_batch):
            if not done:
                argmax_over_a = np.argmax(double_next_q_values_list[index])
                future_reward = next_q_values_list[index][argmax_over_a]
                target = reward + self.gama * future_reward
            else:
                target = reward

            current_q_values = current_q_values_list[index]
            current_q_values[action] = target

            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.squeeze(x_train)
        y_train = np.squeeze(y_train)
        self.model.fit(tf.convert_to_tensor(x_train, tf.float32), tf.convert_to_tensor(y_train, tf.float32),
                       batch_size=32, verbose=0)
        self.update_target_counter += 1

        if self.update_target_counter > self.update_target_network_after and terminal:
            self.update_target_network()

    def save_model(self, env_name, episode_num):
        self.model.save("saved_model/prioritized_dqn_" + str(env_name) + '_' + str(episode_num))

    def load_model(self, env_name, episode_num):
        tf.print("Loading Model ...")
        self.model = tf.keras.models.load_model("saved_model/prioritized_dqn_" + str(env_name) + '_' + str(episode_num))

    def replay_buffer_append(self, transition):
        double_next_q_value = self.model.predict(transition[3])
        next_q_value = self.target_model.predict(transition[3])
        current_q_value = self.model.predict(transition[0])
        argmax_over_a = np.argmax(double_next_q_value[0])
        target = transition[2] + self.gama * next_q_value[0][argmax_over_a]
        td_error = target - current_q_value[0][transition[1]]
        priority = abs(td_error) + self.minimum_priority
        self.replay_memory.push(transition, priority)
