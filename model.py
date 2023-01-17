import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, interior_obs_size, relative_obs_size, hidden_size, output_size, gain=np.sqrt(2)):
        super(PolicyNet, self).__init__()

        self.interior_obs_size = interior_obs_size
        self.relative_obs_size = relative_obs_size
        self.embedded_length = 64

        self.lstm = nn.LSTM(input_size=relative_obs_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.embedded_length)

        self.layer_stack = nn.Sequential(
            nn.Linear(self.interior_obs_size + self.embedded_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Hardswish(),
        )
        self.mu = nn.Linear(hidden_size, output_size)
        self.log_std = nn.Linear(hidden_size, output_size)

        # initialization for lstm.
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # initialization for self.linear.
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.constant_(self.linear.bias, 0)

        # initialization for nn.Sequential.
        for item in self.layer_stack:
            if isinstance(item, nn.Linear):
                nn.init.xavier_uniform_(item.weight, gain=gain)
                nn.init.constant_(item.bias, 0)

        # initialization for output layers.
        nn.init.xavier_uniform_(self.mu.weight, gain=gain)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.xavier_uniform_(self.log_std.weight, gain=gain)
        nn.init.constant_(self.log_std.bias, 0)

    def forward(self, x, y):

        output, (hn, cn) = self.lstm(y)
        z = self.linear(hn.squeeze())

        input_vector = torch.cat([x, z], dim=1)

        tau = self.layer_stack(input_vector)
        mu = torch.tanh(self.mu(tau))
        sigma = torch.exp(self.log_std(tau))

        return mu, sigma


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, value)
        return context, attn


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_v, bias=False)

        self.attention = ScaledDotProductAttention(dim=d_k)
        self.linear = nn.Linear(d_v, d_model, bias=False)
        self.layer_norm = nn.InstanceNorm1d(d_model)

    def forward(self, input_vector):
        residual = input_vector
        query = self.w_q(input_vector)
        key = self.w_k(input_vector)
        value = self.w_v(input_vector)

        context, _ = self.attention(query, key, value)
        output = self.linear(context)

        return self.layer_norm(output + residual)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Hardswish(),
            nn.Linear(hidden_size, input_size),
        )
        self.layer_norm = nn.InstanceNorm1d(input_size)

    def forward(self, input_vector):
        residual = input_vector
        x = self.layer_stack(input_vector)
        output = self.layer_norm(x + residual)

        return output


class ValueNet(nn.Module):
    def __init__(self, input_size, d_model, d_k, d_v, hidden_size):
        super(ValueNet, self).__init__()

        self.linear = nn.Linear(input_size, d_model)
        self.attention_layer = SelfAttentionLayer(d_model=d_model, d_k=d_k, d_v=d_v)
        self.feedforward_layer = PositionWiseFeedForward(input_size=d_model, hidden_size=hidden_size)
        self.layer_stack = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.Hardswish(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_state):

        """
          input_state size (batch_size, num_agents, agent_self_state_length)
          batch_size = total time steps in current buffer, agent_self_state_length = 8 is fixed length.
        """

        x = F.relu(self.linear(input_state))
        x = self.attention_layer(x)
        y = self.feedforward_layer(x)
        values = self.layer_stack(y)

        return values
