import future, sys, os, datetime, argparse
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
nx = 81
dx = uT/(nx-1)
dt = 0.025

c = 1

u = np.zeros(nx)
u[int(0.5/dx):int(1/dx)]=2

# print(np.pad(np.arange(10),[1,1], mode='constant', constant_values=0))

def findiff(u, dx=1, order=1, axis=0):
	if order==2:
		if u.ndim == 1:
			u_padded = np.pad(u, pad_width=[1,1], mode='edge')
			du = (-2*u_padded[1:-1] + u_padded[2:] + u_padded[:-2])/dx**2
			return du
	elif order==1:
		if u.ndim == 1:
			if False:
				u_padded = np.pad(u, pad_width=[1, 1], mode='edge')
				u_ = (u_padded[2:] + u_padded[:-2]) / (2*dx)
			if True:
				u_padded = np.pad(u, pad_width=[1,0], mode='edge')
				du = (u_padded[1:] - u_padded[:-1]) / dx
			return du

print(findiff(u, dx=dx).shape)
for step in range(20):

	# plt.plot(np.linspace(u0, uT, nx), c*dt*findiff(u, dx), c='red')
	u -= c * dt * findiff(u, dx=dx)
	plt.plot(np.linspace(u0,uT,nx), u)
	plt.plot()
	plt.show()