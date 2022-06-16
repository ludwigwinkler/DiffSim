import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

if 0:

    import gym

    env = gym.make("CartPole-v0")
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)

    print()
    env.close()

if 0:
    data_res = 20
    radius = 10
    x = np.linspace(-15, 15, data_res)
    y = np.linspace(-15, 15, data_res)

    X, Y = np.meshgrid(x, y)
    points = Tensor(np.stack((X.flatten(), Y.flatten())).T)
    x_, y_ = points.T

    ### Circular Data ###

    t = torch.atan2(y_, x_)
    radius = points.pow(2).sum(dim=-1).pow(0.5)
    dx_ = -radius * torch.sin(t)
    dy_ = radius * torch.cos(t)
    dx = torch.stack([dx_, dy_]).T  # [2,400] -> [400,2]

    dx = dx.reshape((*X.shape, 2)).numpy()

    assert (
        x_.shape == y_.shape == dx_.shape == dy_.shape
    ), f"{x_.shape=}, {y_.shape=}, {dx_.shape=} {dy_.shape=}"
    plt.quiver(x_, y_, dx_, dy_)
    plt.show()

    ### ForceField ###

    radius = points.pow(2).sum(dim=-1).pow(0.5)
    dx_ = radius * torch.where(y_ < 0, torch.ones_like(y_), -torch.ones_like(y_))
    dy_ = radius * torch.zeros_like(dx_)

    plt.quiver(x_, y_, dx_, dy_)
    plt.show()

g = 10.0
lp = 1.0
mp = 1.0
mk = 0.0
mt = 1.0

a = g / (lp * (4.0 / 3 - mp / (mp + mk)))
A = np.array([[0, 1, 0, 0], [0, 0, a, 0], [0, 0, 0, 1], [0, 0, a, 0]])

# input matrix
b = -1 / (lp * (4.0 / 3 - mp / (mp + mk)))
B = np.array([[0], [1 / mt], [0], [b]])

R = np.eye(1, dtype=int)  # choose R (weight for input)
Q = 5 * np.eye(4, dtype=int)  # choose Q (weight for state)

# get riccati solver
from scipy import linalg

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R), np.dot(B.T, P))


def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)  # u = -Kx
    if u > 0:
        return 1, u  # if force_dem > 0 -> move cart right
    else:
        return 0, u  # if force_dem <= 0 -> move cart left


import gym

env = gym.make("CartPole-v0")
# env.env.seed(1)  # seed for reproducibility
obs = env.reset()

for i in range(1000):
    env.render()

    # get force direction (action) and force value (force)
    action, force = apply_state_controller(K, obs)

    if i < 100:
        action = env.action_space.sample()

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))

    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _ = env.step(action)
    if done and False:
        print(f"Terminated after {i + 1} iterations.")
        break

env.close()
