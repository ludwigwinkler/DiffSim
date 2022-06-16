import future, sys, os, datetime, argparse, copy
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

'''
u: heat/y
x: point in space in [0,2]
nx: grid in space
dx: distance between sampling grid in space
'''
u0, uT = 0, 2
nx = 41
dx = uT/(nx-1)
nu = 0.3

sigma = 0.3
dt = sigma * dx**2


u = np.zeros(nx)
u[int(0.5/dx):int(1/dx)]=2

def finitediff(u, dx=1, order=1, axis=0):
	if order==2:
		if u.ndim == 1:
			u_padded = np.pad(u, pad_width=[1,1], mode='edge')
			du = (-2*u_padded[1:-1] + u_padded[2:] + u_padded[:-2])/dx**2
			return du
	elif order==1:
		if u.ndim == 1:

			'''
			One sided derivative estimate
			'''
			u_padded = np.pad(u, pad_width=[1,0], mode='edge')
			du = (u_padded[1:] - u_padded[:-1]) / dx
	return du

num_steps = 100
for step in range(num_steps):
	plot = True if step%(num_steps//10)==0 else False
	if plot: plt.plot(np.linspace(u0, uT, nx), nu*dt*finitediff(u, dx, order=2) , c='red')


	u += nu*dt * finitediff(u, dx=dx, order=2)
	if plot:
		plt.plot(np.linspace(u0,uT,nx), u)
		plt.show()