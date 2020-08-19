from agents.DQNAgent import DQNAgent
from agents.DoubleDQNAgent import DoubleDQNAgent
from agents.PrioritizedDoubleDQN import PrioritizedDoubleDQNAgent
from metrics.Metric import DQNMetric
import gym
import numpy as np
import datetime
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

"""
TODO: in train method, due to different range of reward in different environments,
        it would be better if normalize rewards to [-1,+1]
"""

LOGGING = True

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class ClassicControl:
    def __init__(self, env_name, dqn_variant='nature', mode='train'):
        """
        Classic Control Class is defined for train and test of the classic control problems of gym
        :param env_name: environment name shows the gym name (e.g. CartPole)
        :param dqn_variant: DQN variant shows the different variants of DQN (e.g. Nature, Dueling, Double)
        :param mode: mode could be train or test
        """
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        if dqn_variant == "nature_dqn":
            self.rl_agent = DQNAgent(self.state_size, self.action_size, environment_type='control',
                                     min_replay_buffer_size=500, update_target_network_after=1000)
        elif dqn_variant == "double_dqn":
            self.rl_agent = DoubleDQNAgent(self.state_size, self.action_size, environment_type='control',
                                           min_replay_buffer_size=500, update_target_network_after=1000)
        elif dqn_variant == "prioritized_dqn":
            self.rl_agent = PrioritizedDoubleDQNAgent(self.state_size, self.action_size, environment_type='control',
                                                      min_replay_buffer_size=500, update_target_network_after=1000)

        self.save_model_frequency = 20
        self.total_episode_counter = 0
        self.total_action_counter = 0
        self.episode_reward = 0

        if LOGGING:
            reward_log_dir = 'logs/gradient_tape/' + dqn_variant + '_' + mode + '/' + current_time + 'reward'
            self.reward_writer = tf.summary.create_file_writer(reward_log_dir)
            self.reward_metric = DQNMetric()

        if mode == 'train':
            self.train()
        elif mode == 'test':
            self.test()

    def train(self):
        while True:
            self.total_episode_counter += 1
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while True:
                self.env.render()
                action = self.rl_agent.act(tf.convert_to_tensor(state, dtype=tf.float32))
                state_next, reward, terminal, _ = self.env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.state_size])
                transition = (state, action, reward, state_next, terminal)
                self.rl_agent.replay_buffer_append(transition)
                self.rl_agent.train(terminal)
                state = state_next

                self.episode_reward += reward
                self.total_action_counter += 1
                if terminal:
                    tf.print(
                        "Episode {} - Accumulated reward {}".format(self.total_episode_counter, self.episode_reward))

                    if self.total_episode_counter % self.save_model_frequency == 0:
                        self.rl_agent.save_model(self.env_name, self.total_episode_counter)

                    if LOGGING:
                        self.reward_metric.update_state(episode_reward=self.episode_reward)
                        with self.reward_writer.as_default():
                            tf.summary.scalar('reward', self.reward_metric.result(), step=self.total_episode_counter)
                        self.reward_metric.reset_states()

                    self.episode_reward = 0
                    break

    def test(self):
        self.rl_agent.load_model(self.env_name, episode_num=90)
        while True:
            self.total_episode_counter += 1
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while True:
                # self.env.render()
                action = self.rl_agent.act(tf.convert_to_tensor(state, dtype=tf.float32))
                state_next, reward, terminal, _ = self.env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.state_size])
                state = state_next

                self.episode_reward += reward
                self.total_action_counter += 1
                if terminal:
                    tf.print(
                        "Episode {} - Accumulated reward {}".format(self.total_episode_counter, self.episode_reward))
                    if LOGGING:
                        self.reward_metric.update_state(episode_reward=self.episode_reward)
                        with self.reward_writer.as_default():
                            tf.summary.scalar('reward', self.reward_metric.result(), step=self.total_episode_counter)
                        self.reward_metric.reset_states()

                    self.episode_reward = 0
                    break


if __name__ == "__main__":
    ClassicControl('CartPole-v0', 'prioritized_dqn', 'train')
