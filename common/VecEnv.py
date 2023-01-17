import numpy as np
from copy import deepcopy


class SequentialVecEnv(object):
    def __init__(self, env_maker, num_envs=1):
        self.num_envs = num_envs

        self.envs = [make_env() for make_env in env_maker]  # a list of environment.
        self.state_space = [env.state.shape for env in self.envs]  # a list of 2 element tuple(num_agents, state_shape).

        # a list for temporally store different shape of state from different environment.
        self.buf_state = [np.zeros((self.state_space[env_index][0], self.state_space[env_index][1]), dtype=float)
                          for env_index in range(num_envs)]
        # a list for temporally store reward from each environment by sequence.
        self.buf_reward = [np.zeros((self.state_space[env_index][0]), dtype=float) for env_index in range(num_envs)]
        # a bool type numpy array for strong done flag from each environment.
        self.buf_terminal = np.zeros((self.num_envs,), dtype=bool)

    def reset(self):
        vec_initial_state = [env.reset() for env in self.envs]
        return vec_initial_state

    def reset_at(self, env_index):
        return self.envs[env_index].reset()

    def step(self, action_matrix):
        for env_idx in range(self.num_envs):
            obs, self.buf_reward[env_idx], self.buf_terminal[env_idx], _ = self.envs[env_idx].step(action_matrix[env_idx])

            if self.buf_terminal[env_idx]:
                # save final observation where user can get it, then reset
                obs = self.envs[env_idx].reset()

            self.buf_state[env_idx] = obs  # env_index as key.

        # return list(self.buf_state), deepcopy(self.buf_reward), np.copy(self.buf_terminal), None
        return deepcopy(self.buf_state), deepcopy(self.buf_reward), np.copy(self.buf_terminal), None
