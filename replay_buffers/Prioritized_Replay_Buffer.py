from replay_buffers.Abstract_Replay_Buffer import AbstractReplayBuffer
import random
from collections import deque


class PrioritizedReplayBuffer(AbstractReplayBuffer):
    def __init__(self, capacity):
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        self.priorities = deque(maxlen=self.replay_buffer.maxlen)

    def push(self, transition_list, priority=None):
        self.num_transitions += 1
        self.replay_buffer.append(transition_list)
        self.priorities.append(priority)

    def sample(self, batch_size):
        return random.choices(population=self.replay_buffer, weights=self.priorities, k=batch_size)

    def get_size(self):
        return self.num_transitions

    def get_capacity(self):
        return self.capacity

    def clear(self):
        self.replay_buffer.clear()
        self.priorities.clear()
        self.num_transitions = 0
