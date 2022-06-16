import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

from numbers import Number

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

def finitedifference(mesh, order=1):
	if mesh.u.dim()==1:
		'''
		One-dimensional mesh
		'''
		if order==1:
			mode=['forward', 'backward'][0]
			if mode=='forward':
				u_padded = mesh.u.reshape(1,1,-1) # [W] -> [BS=1, C_i=1, W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1,0], mode='replicate') #replicate on left side
				du = F.conv1d(u_padded,weight=Tensor([-1,1]).reshape(1,1,-1))/mesh.dx
				du = du.squeeze(0).squeeze(0) # [BS=1, C_i=1, W] -> [W]
				return du
		if order==2:
			u_padded = mesh.u.reshape(1, 1, -1)  # [W] -> [BS=1, C_i=1, W]
			u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1], mode='replicate')  # replicate on left side
			ddu = F.conv1d(u_padded, weight=Tensor([1, -2,1]).reshape(1, 1, -1)) / mesh.dx**2
			ddu = ddu.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
			return ddu

	if mesh.u.dim()==2:
		if order == 1:
			mode = ['forward', 'backward'][0]
			if mode == 'forward':
				u_padded = mesh.u.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1,0,1,0], mode='replicate')  # replicate on left side
				'''
				x axis:
				[0  0]
				[-1 1]
				y axis:
				[0 -1]
				[0  1]
				'''
				weight = Tensor([[[0, 0], [-1, 1]],[[0,-1],[0,1]]]).unsqueeze(1)

				du = F.conv2d(u_padded, weight=weight)
				du = du.squeeze(0).permute(-1,0,1) / Tensor(mesh.dx)
				print(f'{du.shape=}')
				print(f"{mesh.dx=}")
				fig = plt.figure()
				ax = fig.gca(projection='3d')
				X, Y = np.meshgrid(mesh.x[0], mesh.x[1])
				ax.plot_surface(X, Y, du[1].numpy(), cmap=plt.cm.get_cmap('viridis'))
				plt.show()
				exit()
				du = du.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
				return du

class LinearConvection:

	def __init__(self, c):
		self.c = c

	def __call__(self, mesh, dt):

		mesh.u -= self.c * dt * finitedifference(mesh)
		return mesh

class NonLinearConvection:

	def __init__(self):
		pass

	def __call__(self, mesh, dt):

		mesh.u -= mesh.u * dt * finitedifference(mesh, order=1)
		return mesh

class Diffusion:

	def __init__(self, nu):
		self.nu = nu

	def __call__(self, mesh, dt):
		mesh.u += self.nu * dt * finitedifference(mesh, order=2)
		return mesh

class Mesh:

	def __init__(self, space_start, space_end, space_grid_res=None, space_samples=None):

		if isinstance(space_start, Number) and isinstance(space_end, Number):
			'''1D grid'''
			if isinstance(space_grid_res, Number) and space_samples is None:
				self.res = space_grid_res
			if isinstance(space_samples, Number) and space_grid_res is None:
				self.res = (space_end - space_start)/space_samples
		elif isinstance(space_start, list) and isinstance(space_end, list):
			'''2D Grid'''
			pass
		assert len(space_start)==len(space_end)

class Mesh1D:

	def __init__(self):

		self.x_limits = [0,2]
		self.grid_samples = 201
		self.dx = (self.x_limits[1] - self.x_limits[0])/(self.grid_samples-1)
		self.x = torch.linspace(*self.x_limits,steps=self.grid_samples)
		self.u = torch.zeros(self.grid_samples)
		self.u[int(0.5 / self.dx): int(1 / self.dx)]=3
		# self.u +=1

	def plot(self, show=True, color='red'):
		plt.plot(self.x, self.u, color=color)
		plt.grid()
		if show: plt.show()

class Mesh2D:

	def __init__(self):
		self.x_limits = [[0, 2],[0,2]]
		self.grid_samples = [201,201]
		self.dx = [(self.x_limits[0,1] - self.x_limits[0,0]) / (self.grid_samples[0] - 1),
			   (self.x_limits[1,1] - self.x_limits[1,0]) / (self.grid_samples[1] - 1)]
		self.x = torch.linspace(*self.x_limits, steps=self.grid_samples)
		self.u = torch.zeros(self.grid_samples)
		self.u[int(0.5 / self.dx): int(1 / self.dx)] = 3


	def plot(self, show=True, color='red'):
		plt.plot(self.x, self.u, color=color)
		plt.grid()
		if show: plt.show()


class CFD:

	def __init__(self, mesh, forces=None):
		'''

		:param mesh: discretized u
		:param forces: list of forces that act on the mesh like convection, diffusion etc
		'''
		self.mesh = mesh
		self.dt = 0.01**2
		# self.forces = [LinearConvection(c=.25)]
		self.forces = [Diffusion(nu=.3)]
		# self.forces = [Diffusion(nu=.3), LinearConvection(c=3.)]
		# self.forces = [Diffusion(nu=.3), NonLinearConvection()]
		# self.forces = [NonLinearConvection()]

	def solve(self, t):

		self.num_plots=10
		cmap = plt.cm.get_cmap('Spectral')
		colors = cmap(np.linspace(0, 1, self.num_plots)).tolist() # [c1, c2, c3, ... c11]
		plot_every = t//len(colors)
		for step in range(t):
			if step%plot_every==0:
				print(f"{step=}")
				mesh.plot(show=False, color=colors.pop(0))
			for force in self.forces:
				self.mesh = force(self.mesh, self.dt)

		plt.show()

if __name__=='main':

	mesh = Mesh1D()
	cfd = CFD(mesh)
	cfd.solve(t=500)