import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence

from model import PolicyNet, ValueNet
from rolloutbuffer import Buffer


class MAPPO(object):
    def __init__(self, env, num_steps, alpha, beta, max_std, clip, gamma, k_epochs, use_bcloss, bcloss_weight,
                 use_init_model=False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = env  # must be a vectorized environment.
        self.num_envs = self.env.num_envs

        self.env_attr = self.env.attributes
        self.num_agents = [num_agents for num_agents, _, _ in self.env_attr]
        self.state_dim = [state_dim for _, state_dim, _ in self.env_attr]
        self.action_dim = 2
        self.num_steps = num_steps  # the number of steps need to be sampled for each environment in each iteration.

        self.max_state_length = max(self.state_dim)
        self.max_num_agents = max(self.num_agents)

        # initialize rollout buffer for the each environment in the vectorized wrapper.
        self.buffer = [Buffer(self.num_steps, self.num_agents[env_id], self.state_dim[env_id], self.action_dim)
                       for env_id in range(self.num_envs)]

        # Declare the actor and critic network according to the reward type.
        self.actor = PolicyNet(interior_obs_size=8, relative_obs_size=2, hidden_size=256, output_size=2).to(self.device)
        self.critic = ValueNet(input_size=8, d_model=32, d_k=64, d_v=64, hidden_size=128).to(self.device)

        # create the optimizer for actor and critic network.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=beta)

        self.max_std = max_std
        self.gamma = gamma
        self.clip = clip
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.use_bcloss = use_bcloss
        self.bcl_weight = bcloss_weight
        self.expert_data = None

        # If use the pretrained model, load the model .
        if use_init_model:
            self.load_initial_model()

        # If use the behavior cloning loss, load the prepared dataset.
        if self.use_bcloss:
            self.load_expert_dataset()

    def get_action(self, interior_state, relative_state):
        with torch.no_grad():
            # query actor to obtain the parameters of action distribution.
            mu, sigma = self.actor(interior_state, relative_state)

            # restrict the standard deviation within a suitable range for ability of exploration.
            sigma = torch.clamp(sigma, min=1e-2, max=self.max_std)

            # create a distribution according to the output of neural network.
            distribution = Normal(mu, sigma)

            # sample action from the distribution and calculate the log probability.
            sampled_action = distribution.sample()
            log_prob = distribution.log_prob(sampled_action).sum(dim=1)

        return sampled_action, log_prob

    def rollout(self):

        states = self.env.reset()
        for step in range(self.num_steps):

            actions = []
            log_probs = []
            for env_index in range(self.num_envs):  # iterate through all envs to get action from current policy.

                num_agents, obs_length = states[env_index].shape[0], states[env_index].shape[1]
                observation = torch.FloatTensor(states[env_index]).to(self.device)

                interior_obs = observation[0:num_agents, 0:8]
                relative_obs = observation[0:num_agents, 8:obs_length].reshape(num_agents, num_agents - 1, 2)

                action, log_prob = self.get_action(interior_obs, relative_obs)
                action = action.cpu().numpy()

                actions.append(action)
                log_probs.append(log_prob)

            next_states, rewards, terminals, infos = self.env.step(actions)  # step the env by performing the actions.

            for i in range(self.num_envs):  # save the transitions to buffers.
                if terminals[i]:
                    self.buffer[i].push(states[i], actions[i], log_probs[i], rewards[i], terminals[i])
                else:
                    self.buffer[i].push(states[i], actions[i], log_probs[i], rewards[i], terminals[i])

            states = next_states  # update the states for all environment.

    def merge_buffer(self):

        batch_observation, seq_length = [], []  # for agent local observation, seq_length is used for pack&pad sequence.
        batch_global_state = []  # a list for store global state in every time step.
        batch_action = []
        batch_log_prob = []
        batch_return = []

        rewards_info = []  # used to store the episodic reward information for data within current buffer.
        for env_id in range(self.num_envs):

            state, action, log_prob, reward, done = self.buffer[env_id].get_data()
            rewards_info.append(np.sum(reward) / (self.num_steps / 300))

            with torch.no_grad():
                batch_global_state.append(torch.FloatTensor(state[:, :, 0:8]).to(self.device))
                batch_action.append(torch.FloatTensor(action.reshape(-1, self.action_dim)).to(self.device))
                batch_log_prob.append(torch.cat(log_prob, dim=0))

            num_time_steps, num_agents, obs_length = state.shape[0], state.shape[1], state.shape[2]
            seq_length.append(np.full(num_time_steps * num_agents, num_agents - 1))

            if num_agents < self.max_num_agents:
                pad = np.zeros((num_time_steps, num_agents, self.max_state_length - obs_length))
                padded_obs = np.concatenate([state, pad], axis=2)
                batch_observation.append(padded_obs.reshape(-1, padded_obs.shape[-1]))

            else:  # no need to pad, directly flat the state and add to the list.
                batch_observation.append(state.reshape(-1, state.shape[-1]))

            # calculate the episodic returns for every time step in current buffer.
            returns = []
            discounted_rewards = 0
            for j in reversed(range(done.shape[0])):

                if done[j]:
                    discounted_rewards = 0

                r = reward[j]
                discounted_rewards = r + discounted_rewards * self.gamma
                returns.insert(0, discounted_rewards)

            batch_return.append(np.asarray(returns))

        # split agent interior obs and agent relative obs, and put relative obs as packed padded sequence.
        batch_observation = np.vstack(batch_observation)
        seq_length = np.concatenate(seq_length, axis=0)

        with torch.no_grad():
            batch_action = torch.cat(batch_action, dim=0)
            batch_log_prob = torch.cat(batch_log_prob, dim=0)

            interior_obs = torch.FloatTensor(batch_observation[:, 0:8]).to(self.device)
            relative_obs = torch.FloatTensor(batch_observation[:, 8:self.max_state_length]).to(self.device)
            relative_obs = relative_obs.reshape(relative_obs.size(0), self.max_num_agents - 1, 2)
            relative_obs = pack_padded_sequence(relative_obs, seq_length, batch_first=True)

        return interior_obs, relative_obs, batch_global_state, batch_action, batch_log_prob, batch_return, rewards_info

    def update(self, baseline):

        interior_obs, relative_obs, global_state, action, old_log_prob, batch_return, episodic_rewards = \
            self.merge_buffer()

        # calculate the advantages and the and shape the target values to proper dimension.
        with torch.no_grad():

            advantages = []
            target_values = []
            for i in range(self.num_envs):
                returns = torch.FloatTensor(batch_return[i]).to(self.device)
                target_values.append(torch.flatten(returns))

                v = self.critic(global_state[i])
                v = v.flatten().squeeze()

                adv = returns.flatten() - v
                adv = (adv - adv.mean()) / (adv.std() + 1e-10)
                advantages.append(adv)

            advantages = torch.cat(advantages, dim=0)
            target_values = torch.cat(target_values, dim=0)

        # compute the performance ratio.
        pr = np.mean(np.asarray(episodic_rewards, dtype=float)) / baseline

        # perform the PPO update.
        for _ in range(self.k_epochs):

            # get new mu, log_std, predicted_action, and predict_values.
            mu, sigma = self.actor(interior_obs, relative_obs)

            # clamp the standard deviation.
            sigma = torch.clamp(sigma, min=0.01, max=self.max_std)

            # create a new distribution.
            new_distribution = Normal(mu, sigma)
            entropy_regularizer = torch.mean(new_distribution.entropy())
            new_log_prob = new_distribution.log_prob(action).sum(dim=1)

            # calculate the actor loss and critic loss.
            ratio = torch.exp(new_log_prob.squeeze() - old_log_prob)
            s1 = ratio * advantages
            s2 = (torch.clamp(ratio, 1 - self.clip, 1 + self.clip)) * advantages

            # compute the policy loss according to the agent setup.
            if pr > 1.5:
                torch.autograd.set_detect_anomaly(True)

                # sample a batch of state action pair from the expert dataset.
                index = torch.randint(self.expert_data.shape[0], (action.shape[0], ))
                samples = torch.index_select(self.expert_data, 0, index)
                sorted_samples = samples[samples[:, -1].argsort(descending=True)]  # sort by the value of last column.

                # extract the necessary information from the expert dataset.
                sampled_states = sorted_samples[:, 0:self.max_state_length]
                target_actions = sorted_samples[:, -3:-1].to(self.device)
                seq_length = sorted_samples[:, -1]  # the last column store the seq_length = num_agent - 1.

                # convert the samples to tensor object and pack as a padded sequence for feed RNN.
                interior_obs = torch.FloatTensor(sampled_states[:, 0:8]).to(self.device)
                relative_obs = torch.FloatTensor(sampled_states[:, 8:self.max_state_length]).to(self.device)
                relative_obs = relative_obs.reshape(relative_obs.size(0), self.max_num_agents - 1, 2)
                relative_obs = pack_padded_sequence(relative_obs, seq_length, batch_first=True)

                predicted_action, _ = self.actor(interior_obs, relative_obs)
                bc_loss = F.mse_loss(predicted_action, target_actions, reduction='mean')
                policy_loss = (-1 * torch.min(s1, s2)).mean() - 0.01 * entropy_regularizer + self.bcl_weight * bc_loss

            else:
                policy_loss = (-1 * torch.min(s1, s2)).mean() - 0.01 * entropy_regularizer

            # compute the critic loss.
            predict_values = []
            for i in range(self.num_envs):
                values = self.critic(global_state[i]).flatten().squeeze()
                predict_values.append(values)

            predict_values = torch.cat(predict_values, dim=0)
            critic_loss = F.mse_loss(predict_values, target_values, reduction='mean')

            # perform the actor network update
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 200)
            self.actor_optimizer.step()

            # perform the critic network update
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 500)
            self.critic_optimizer.step()

        return [episodic_rewards, pr]

    def clear_buffer(self):
        for i in range(self.num_envs):
            self.buffer[i].clear_buffer()

    def save_model(self, checkpoint):
        torch.save(self.actor, './trained_model/general_policy_{}.pth'.format(str(checkpoint)))
        torch.save(self.critic, './trained_model/general_critic_{}.pth'.format(str(checkpoint)))

    def load_model(self, path, checkpoint):
        self.actor = torch.load(path + './trained_model/generic_policy_{}.pth'.format(str(checkpoint)))
        self.critic = torch.load(path + './trained_model/generic_critic_{}.pth'.format(str(checkpoint)))

    def load_initial_model(self):
        self.actor.load_state_dict(torch.load(r"./trained_model/initial_actor_weight.pth"))
        self.critic.load_state_dict(torch.load(r"./trained_model/initial_critic_weight.pth"))

    def load_expert_dataset(self):
        self.expert_data = np.load(r"./dataset/expert_dataset.npy")
        self.expert_data = torch.tensor(self.expert_data, dtype=torch.float, requires_grad=False)
