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
nx = 101
dx = uT/(nx-1)
dt = 0.025
c = 1

print(f"{nx=} {dx=} {dt=}")

# exit()

sigma_max = 0.5

linear_conv = True

if linear_conv:
	CFL = (c * dt)/dx

	old_dt = dt
	if CFL>=sigma_max:
		while CFL>=sigma_max:
			dt*=0.95
			CFL = (c * dt) / dx

		print(f"{old_dt=} {dt=}")
	assert CFL<=sigma_max, f"{CFL=} <= {sigma_max}"


u = np.ones(nx)
u[int(0.5/dx):int(1/dx)]=2

def findiff(u, dx=1, order=1, axis=0):
	if order==2:
		if u.ndim == 1:
			u_padded = np.pad(u, pad_width=[1,1], mode='edge')
			du = (-2*u_padded[1:-1] + u_padded[2:] + u_padded[:-2])/dx**2
			return du
	elif order==1:
		if u.ndim == 1:
			if False:
				'''
				Two sided derivative estimate
				'''
				u_padded = np.pad(u, pad_width=[1, 1], mode='edge')
				du = (u_padded[2:] - u_padded[:-2]) / (2*dx)
			else:
				'''
				One sided derivative estimate
				'''
				u_padded = np.pad(u, pad_width=[1,0], mode='edge')
				du = (u_padded[1:] - u_padded[:-1]) / dx
	# print(f"{u.shape=} {du.shape}")
	# exit()
	return du

print(findiff(u, dx=dx).shape)
for step in range(51):
	plot = True if step//10==0 else False
	if plot: plt.plot(np.linspace(u0, uT, nx), c*dt*findiff(u, dx), c='red')

	if not linear_conv:
		dt = 0.95*sigma_max*dx/u.max()

	if linear_conv:
		u -= c* dt * findiff(u, dx=dx)
	elif not linear_conv:
		u -= u * dt * findiff(u, dx=dx)
	if plot:
		plt.plot(np.linspace(u0,uT,nx), u)
		plt.show()