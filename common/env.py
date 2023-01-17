import gym
import math
import numpy as np
from random import random


class Custom_Environment(gym.Env):
    def __init__(self, domain, num_agents, step_length=0.1, max_steps=200, reward_type="LR"):

        self.num_agents = num_agents  # the number of agents in the environment.
        self.domain = domain  # specify the domain.
        self.rd = math.sqrt(self.domain.area / self.num_agents)  # the rd distance for converge.
        self.t = step_length  # time spacing.
        self.reward_type = reward_type

        # maximum simulation length.
        self.max_steps = max_steps  # maximum simulation length.
        self.current_steps = 0
        self.done = False  # flag for achieve maximum simulation step length.
        self.completion = False  # flag for achieve termination.

        # initializing states, features = [pi_x, pi_y, vi_x, vi_y, hi_x, hi_y, distance, sign, (pj_x, pj_y,....)].
        self.state = np.zeros((self.num_agents, 8 + 2 * (self.num_agents - 1)), dtype=float)

    # calculate the reward for current states for each agent.
    def calculate_reward(self):

        # calculate the potential for each agents.
        potential = np.zeros(self.num_agents, dtype=float)
        for m in range(self.num_agents):

            d = self.state[m, 6]  # domain distance.
            sign = self.state[m, 7]  # sign indicator

            # calculate the vehicle-domain potential.
            if (sign * d) <= -0.5 * self.rd:
                VH = 0
            else:
                VH = 0.5 * (((sign * d) + (0.5 * self.rd)) ** 2)

            # calculate the inter-vehicle potential.
            VI = 0  # initializing inter-vehicle potential for vehicle m.
            cursor = 8
            for n in range(self.num_agents - 1):

                p_mn = self.state[m, cursor:cursor + 2]
                norm = np.linalg.norm(p_mn, ord=2, keepdims=True)

                if norm < self.rd:
                    V_mn = (1 / 2) * (math.pow((norm - self.rd), 2))
                else:
                    V_mn = 0

                VI = VI + V_mn
                cursor = cursor + 2

            potential[m] = (2 * VH + VI)

        reward = np.ones(self.num_agents, dtype=float)  # calculate the potential for each agents.

        # assign reward according to the reward type.
        if self.reward_type == "LR":

            # local reward use the negative of agent's own potential.
            for k in range(self.num_agents):
                indicator = self.state[k, 7]

                if indicator < 0:
                    reward[k] = -0.5 * potential[k]
                else:
                    reward[k] = -20

        elif self.reward_type == "GR":

            # global reward use the negative of system potential(sum of all agent's potential).
            flag = 0
            for k in range(self.num_agents):
                indicator = self.state[k, 7]

                if indicator < 0:
                    flag = flag + 1
                else:
                    pass

            if flag != self.num_agents:
                reward = -20 * reward

            else:
                reward = reward * (-0.5 * np.sum(potential))

        else:
            reward = None

        return reward

    def step(self, action_matrix):
        self.current_steps = self.current_steps + 1
        new_states = np.zeros((self.num_agents, 8 + 2 * (self.num_agents - 1)), dtype=float)

        # calculate the new vehicle position.
        for i in range(self.num_agents):
            u_i = action_matrix[i]
            p_i = self.state[i, 0:2]
            v_i = self.state[i, 2:4]

            new_p_i_x = p_i[0] + v_i[0] * self.t + (1 / 2) * u_i[0] * math.pow(self.t, 2)
            new_p_i_y = p_i[1] + v_i[1] * self.t + (1 / 2) * u_i[1] * math.pow(self.t, 2)

            new_v_i_x = v_i[0] + u_i[0] * self.t
            new_v_i_y = v_i[1] + u_i[1] * self.t

            new_states[i, 0:2] = np.array([new_p_i_x, new_p_i_y])  # set new position vector for all agents
            new_states[i, 2:4] = np.array([new_v_i_x, new_v_i_y])  # set new velocity vector for all agents

        # calculate the new agent's own features, (position, velocity, normalized direction, distance, sign).
        for j in range(self.num_agents):
            h, d, sign, _ = self.domain.measure_distance(new_states[j, 0:2])

            new_states[j, 4:6] = h / d  # the normalized direction of the vehicle to the domain boundary.
            new_states[j, 6] = d  # the unsigned distance of vehicle to the boundary.
            new_states[j, 7] = sign  # the sign indicator representing the agent is inside or outside the domain.

        # set the inter_vehicle distance for all agents
        for k in range(self.num_agents):
            s_k = self.embed_inter_vehicle_distance(new_states, k).reshape((self.num_agents - 1) * 2)
            new_states[k, 8:new_states.shape[1]] = s_k

        self.state = new_states
        reward = self.calculate_reward()

        if self.current_steps >= self.max_steps:
            self.done = True
        else:
            self.done = False

        place_holder = None

        return self.state, reward, self.done, place_holder

    def reset(self):
        self.current_steps = 0
        self.done = False
        initial_state = np.zeros((self.num_agents, 8 + 2 * (self.num_agents - 1)), dtype=float)

        # randomize the initial agent's spacing and height position.
        height_mean = -12
        min_height = height_mean + 2
        max_height = height_mean - 2

        min_spacing = self.rd - 1
        max_spacing = self.rd + 1

        random_value = random()
        rand_height = min_height + (random_value * (max_height - min_height))
        rand_spacing = min_spacing + (random_value * (max_spacing - min_spacing))

        # set initial position for agents, separated into odd and even case.
        if self.num_agents % 2 == 0:
            factor = self.num_agents / 2
            a_0 = np.array([(-1 * (factor - 0.5) * rand_spacing), rand_height])
            initial_state[0, 0:2] = a_0

            for i in range(1, self.num_agents):
                a_i = np.array([a_0[0] + (i * rand_spacing), rand_height])
                initial_state[i, 0:2] = a_i

        else:
            factor = math.floor(self.num_agents / 2)
            a_0 = np.array([(-1 * factor * rand_spacing), rand_height])
            initial_state[0, 0:2] = a_0

            for i in range(1, self.num_agents):
                a_i = np.array([a_0[0] + (i * rand_spacing), rand_height])
                initial_state[i, 0:2] = a_i

        # set the initial agent's own features, (position, velocity, normalized_direction, distance, sign).
        for j in range(self.num_agents):
            h, d, sign, _ = self.domain.measure_distance(initial_state[j, 0:2])

            initial_state[j, 4:6] = h / d  # the normalized direction of the vehicle to the domain boundary.
            initial_state[j, 6] = d  # the unsigned distance of vehicle to the domain boundary.
            initial_state[j, 7] = sign  # the sign indicator representing the agent is inside or outside the domain.

        # set initial inter-vehicle distance.
        for k in range(self.num_agents):
            s_k = self.embed_inter_vehicle_distance(initial_state, k).flatten()
            initial_state[k, 8:initial_state.shape[1]] = s_k

        self.state = initial_state
        return self.state

    def render(self, mode='None', close=False):
        raise NotImplementedError

    # calculate the inter_vehicle distance according to the given index
    def embed_inter_vehicle_distance(self, states, vehicle_index):
        s = []
        p_i = states[vehicle_index, 0:2]
        for j in range(self.num_agents):
            if j != vehicle_index:
                p_j = states[j, 0:2]
                p_ij = p_i - p_j
                s.append(p_ij)

        # sort inter_vehicle distance in ascending order.
        for i in range(len(s) - 1):
            for j in range(0, len(s) - i - 1):
                if np.linalg.norm(s[j]) < np.linalg.norm(s[j + 1]):
                    a = s[j]
                    b = s[j + 1]
                    s[j], s[j + 1] = b, a

        s = np.asarray(s, dtype=np.float32)
        return s
