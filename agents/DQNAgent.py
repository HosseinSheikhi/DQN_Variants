from agents.DQNBase import DQNBase
import tensorflow as tf
import numpy as np


class DQNAgent(DQNBase):
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

        super(DQNAgent, self).__init__(action_size, environment_type,
                                       replay_buffer_size=replay_buffer_size,
                                       min_replay_buffer_size=min_replay_buffer_size,
                                       mini_batch_size=mini_batch_size,
                                       gama=gama,
                                       learning_rate=learning_rate,
                                       epsilon_min=epsilon_min,
                                       update_target_network_after=update_target_network_after)

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

        x_train = []
        y_train = []

        for index, (current_state, action, reward, next_state, done) in enumerate(data_in_mini_batch):
            if not done:
                future_reward = np.max(next_q_values_list[index])
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

    def replay_buffer_append(self, transition):
        self.replay_memory.push(transition)
