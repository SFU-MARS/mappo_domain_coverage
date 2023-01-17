import torch
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from common.env_config import square, hexagon, polygon_1, polygon_2, polygon_3, polygon_4
from common.env import Custom_Environment

""" 
  you can directly run this script to observe the behavior of the well trained neural network policy.
"""


def get_random_env():
    flag = rand.randint(0, 5)

    if flag == 0:
        return Custom_Environment(**square)

    elif flag == 1:
        return Custom_Environment(**hexagon)

    elif flag == 2:
        return Custom_Environment(**polygon_1)

    elif flag == 3:
        return Custom_Environment(**polygon_2)

    elif flag == 4:
        return Custom_Environment(**polygon_3)

    elif flag == 4:
        return Custom_Environment(**polygon_4)


# ======================================================================================================================
# randomly select an environment and play an episode of game.
env = get_random_env()
num_agents = env.num_agents

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = torch.load(r"./trained_model/generic_policy_1000.pth").to(device)
critic = torch.load(r"./trained_model/generic_critic_1000.pth").to(device)
print(policy)
print(critic)

# ======================================================================================================================
state_record = []
reward_record = []
value_record = []
returns_record = []

state = env.reset()
time_steps = 0
done = False
while not done:
    input_state = torch.FloatTensor(state).to(device)
    agent_self_obs = input_state[0:num_agents, 0:8]
    agent_relative_obs = input_state[0:num_agents, 8:input_state.size(1)].reshape(num_agents, num_agents - 1, 2)

    action, _ = policy(agent_self_obs, agent_relative_obs)
    action = action.detach().cpu().numpy()

    value = critic(input_state[0:num_agents, 0:8].unsqueeze(dim=0))
    value_record.append(torch.mean(value.squeeze()).item())

    next_state, reward, done, _ = env.step(action)

    state_record.append(state)
    reward_record.append(reward)

    state = next_state

discounted_rewards = 0
for j in reversed(range(len(reward_record))):
    r = reward_record[j]
    discounted_rewards = r + discounted_rewards * 0.99
    returns_record.insert(0, np.mean(discounted_rewards))

# ======================================================================================================================
# get the initial position.
state_record = np.stack(state_record)
reward_record = np.stack(reward_record)
print(state_record.shape)
print(reward_record.shape)

initial_x = state_record[0, 0:num_agents, 0]
initial_y = state_record[0, 0:num_agents, 1]

# for plotting the domain boundary
bx, by = env.domain.shape.exterior.xy
color = 'b'

# initial plot
fig, ax = plt.subplots()
ax.plot(bx, by, c="red")
ax.scatter(initial_x, initial_y, color=color, s=5)
ax.set_title("frame {}".format(0))
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)


def animation_frame(i):
    ax.clear()
    new_x = state_record[i, 0:num_agents, 0]
    new_y = state_record[i, 0:num_agents, 1]
    new_scatter = ax.scatter(new_x, new_y, color=color, s=5)
    ax.set_title("frame {}".format(i + 1))
    ax.plot(bx, by, c="red")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    return new_scatter,


total_frames = state_record.shape[0]
animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, total_frames, 1), interval=100, repeat=False)
plt.show()

t = np.arange(len(returns_record))
plt.figure(0)
plt.plot(t, returns_record, linewidth=0.5, color='blue')
plt.plot(t, value_record, linewidth=0.5, color='red')
plt.title(label="Critic Loss")
plt.grid()
plt.show()
