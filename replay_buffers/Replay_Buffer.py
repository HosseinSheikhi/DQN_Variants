import random
from replay_buffers.Abstract_Replay_Buffer import AbstractReplayBuffer


class ReplayBuffer(AbstractReplayBuffer):
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__(capacity)

    def push(self, transition_list, priority=None):
        self.num_transitions += 1
        self.replay_buffer.append(transition_list)

    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def get_size(self):
        return self.num_transitions

    def get_capacity(self):
        return self.capacity

    def clear(self):
        self.replay_buffer.clear()
        self.num_transitions = 0
