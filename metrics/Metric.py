import tensorflow as tf


class DQNMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dqn_metric'):
        """
        All state variables should be created in this method by calling self.add_weight()
        :param name:
        """
        super(DQNMetric, self).__init__(name=name)
        self.episode_reward = self.add_weight(name='reward', initializer='zero')

    def update_state(self, episode_reward):
        """
        Has all updates to the state variables like: self.var.assign_add(...)
        In this implementation, this method calls just once at the each episode
        :param episode_reward: Total accumulated reward in each episode
        :return: None
        """
        self.episode_reward.assign_add(episode_reward)

    def result(self):
        """
        Computes and returns a value for the metric from the state variables.
        :return: returns the episode reward
        """
        return self.episode_reward

    def reset_states(self):
        self.episode_reward.assign(0)
