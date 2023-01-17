import numpy as np


class Buffer:
    def __init__(self, max_capacity, num_agents, state_dim, action_dim):

        self.max_capacity = max_capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.buf_state = np.zeros((self.max_capacity, self.num_agents, self.state_dim), dtype=float)
        self.buf_action = np.zeros((self.max_capacity, self.num_agents, self.action_dim), dtype=float)
        self.buf_log_prob = []  # only log_probability is store in a list, each component has torch.Size([num_agents]).
        self.buf_reward = np.zeros((self.max_capacity, self.num_agents), dtype=float)
        self.buf_done = np.zeros(self.max_capacity, dtype=bool)

        self.position_pointer = 0

    def push(self, state, action, log_prob, reward, done):

        if self.position_pointer < self.max_capacity:
            self.buf_state[self.position_pointer] = state
            self.buf_action[self.position_pointer] = action
            self.buf_log_prob.append(log_prob)
            self.buf_reward[self.position_pointer] = reward
            self.buf_done[self.position_pointer] = done

            self.position_pointer = self.position_pointer + 1
        else:
            raise IndexError('exceed maximum buffer capacity')

    def get_data(self):
        return self.buf_state, self.buf_action, self.buf_log_prob, self.buf_reward, self.buf_done

    def clear_buffer(self):
        self.buf_state = np.zeros((self.max_capacity, self.num_agents, self.state_dim), dtype=float)
        self.buf_action = np.zeros((self.max_capacity, self.num_agents, self.action_dim), dtype=float)
        self.buf_log_prob = []
        self.buf_reward = np.zeros((self.max_capacity, self.num_agents), dtype=float)
        self.buf_done = np.zeros(self.max_capacity, dtype=bool)

        self.position_pointer = 0
