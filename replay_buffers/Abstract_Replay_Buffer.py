import abc
from collections import deque


class AbstractReplayBuffer():
    """
    Abstract Base Class for Replay Buffer for DQN and its variants
    """

    def __init__(self, capacity):
        """
        :param capacity: maximum capacity of replay buffer
        """
        self.capacity = capacity
        self.replay_buffer = deque(maxlen=capacity)
        self.num_transitions = 0

    @abc.abstractmethod
    def push(self, transition_list, priority=None):
        """
        appends a transition to the replay buffer
        :param transition_list: a transition list contains [state, action, next_state, reward, done]
        :param priority: priority correspond to the transition_list. just uses for prioritized replay buffer
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        """
        :param batch_size: size of the desire sample
        :return: a mini_batch by size = batch_size  of transitions from replay buffer
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self):
        """
        :return: number of transitions in replay buffer
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_capacity(self):
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError
