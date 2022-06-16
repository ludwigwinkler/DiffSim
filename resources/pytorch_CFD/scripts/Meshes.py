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

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

class Mesh1D:

	def __init__(self):
		self.x_limits = [0, 2]
		self.grid_samples = 201
		self.dx = (self.x_limits[1] - self.x_limits[0]) / (self.grid_samples - 1)
		self.x = torch.linspace(*self.x_limits, steps=self.grid_samples)
		self.u = torch.zeros(self.grid_samples)
		index = lambda val: int((val - self.x_limits[0]) / (self.x_limits[1] - self.x_limits[0]) * self.grid_samples)
		self.u[index(0.5): index(1.)] = 3

	# self.u +=1

	def plot(self, show=True, color='red'):
		plt.plot(self.x, self.u, color=color)
		plt.grid()
		if show: plt.show()

class Mesh2D:

	def __init__(self, x_limits, grid_samples, dx, x, u):
		self.xlimits = x_limits
		self.grid_samples = grid_samples
		self.dx = dx
		self.x = x
		self.u = u

	def plot(self, show=True, color='red'):
		'''
		 _ _ _ _
		|       |
		|       | y: dim=0
		|_ _ _ _|
		   x: dim=1
		Numpy is column first ordering
		We want first dimension on the horizontal axis
		'''
		'''
		Meshgrid takes x axis of grid as first argument and y axis as second argument
		'''
		X, Y = np.meshgrid(self.x[1], self.x[0])
		print(f'{X.shape=}, {Y.shape=}, {self.u.shape=}')
		contour = plt.contourf(X, Y, self.u.numpy(), cmap=plt.cm.get_cmap('viridis'))
		# contour = plt.contourf(self.u.numpy(), cmap=plt.cm.get_cmap('viridis'))
		cbar = plt.colorbar(contour)
		plt.grid()
		plt.xlabel('x', fontsize=20)
		plt.ylabel('y', fontsize=20)
		if show: plt.show(); exit(f'Plot Mesh2d')

	def plot3d(self, show=True):
		fig = plt.figure(figsize=(11, 7), dpi=100)

		ax = fig.gca(projection='3d')
		ax.view_init(30, 235)
		X, Y = np.meshgrid(self.x[1], self.x[0])
		surf = ax.plot_surface(X, Y, self.u.numpy(), cmap=plt.cm.get_cmap('viridis'))

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		plt.show()

class ConvectDiffusMesh(Mesh2D):

	def __init__(self):
		x_limits = [[-1, 3], [-1, 2]]  # first dim vertical, second dim horizontal [y_limit, x_limit]
		grid_samples = [301, 201]
		dx = [(x_limits[0][1] - x_limits[0][0]) / (grid_samples[0] - 1),
		      (x_limits[1][1] - x_limits[1][0]) / (grid_samples[1] - 1)]
		x = [torch.linspace(start=x_limits[0][0], end=x_limits[0][1], steps=grid_samples[0]),
		     torch.linspace(start=x_limits[1][0], end=x_limits[1][1], steps=grid_samples[1])]
		'''
		x on vertical axis, y on horizontal axis -> during plotting
		'''
		u = torch.zeros(*grid_samples)
		index = lambda val, dim: int((val - x_limits[dim][0]) / (x_limits[dim][1] - x_limits[dim][0]) * grid_samples[dim])
		u[index(0, 0): index(0.5, 0),
		index(0, 1):index(0.5, 1)] = 3

		Mesh2D.__init__(self, x_limits, grid_samples, dx, x, u)

class LaplaceMesh(Mesh2D):

	def __init__(self):
		x_limits = [[0, 1], [0, 2]]  # first dim vertical, second dim horizontal [y_limit, x_limit]
		grid_samples = [31, 31]
		dx = Tensor([(x_limits[0][1] - x_limits[0][0]) / (grid_samples[0] - 1),
			     (x_limits[1][1] - x_limits[1][0]) / (grid_samples[1] - 1)]).squeeze()
		x = [torch.linspace(start=x_limits[0][0], end=x_limits[0][1], steps=grid_samples[0]),
		     torch.linspace(start=x_limits[1][0], end=x_limits[1][1], steps=grid_samples[1])]
		'''
		x on vertical axis, y on horizontal axis -> during plotting
		'''
		u = torch.zeros(*grid_samples)

		Mesh2D.__init__(self, x_limits, grid_samples, dx, x, u)
		self = LaplaceBoundaryCondition()(self)

class PoissonMesh(Mesh2D):

	def __init__(self):
		x_limits = [[0, 1], [0, 2]]  # first dim vertical, second dim horizontal [y_limit, x_limit]
		grid_samples = [100, 100]
		dx = Tensor([(x_limits[0][1] - x_limits[0][0]) / (grid_samples[0] - 1),
			     (x_limits[1][1] - x_limits[1][0]) / (grid_samples[1] - 1)]).squeeze()
		x = [torch.linspace(start=x_limits[0][0], end=x_limits[0][1], steps=grid_samples[0]),
		     torch.linspace(start=x_limits[1][0], end=x_limits[1][1], steps=grid_samples[1])]
		'''
		x on vertical axis, y on horizontal axis -> during plotting
		'''
		u = torch.zeros(*grid_samples)

		Mesh2D.__init__(self, x_limits, grid_samples, dx, x, u)
		self.b = torch.zeros_like(self.u)
		self.b[int(0.25 * self.b.shape[0]), int(0.25 * self.b.shape[1])] = 100
		self.b[int(0.75 * self.b.shape[0]), int(0.75 * self.b.shape[1])] = -100

class NavierStokesMesh:

	def __init__(self):
		self.x_limits = [[0, 2], [0, 2]]  # first dim vertical, second dim horizontal [y_limit, x_limit]
		self.grid_samples = [41, 41]
		self.dx = Tensor([(self.x_limits[0][1] - self.x_limits[0][0]) / (self.grid_samples[0] - 1),
			     (self.x_limits[1][1] - self.x_limits[1][0]) / (self.grid_samples[1] - 1)]).squeeze()
		self.x = [torch.linspace(start=self.x_limits[0][0], end=self.x_limits[0][1], steps=self.grid_samples[0]),
		     torch.linspace(start=self.x_limits[1][0], end=self.x_limits[1][1], steps=self.grid_samples[1])]

		'''
		x on vertical axis, y on horizontal axis -> during plotting
		'''
		self.u = torch.zeros(*self.grid_samples)
		self.v = torch.zeros(*self.grid_samples)
		self.p = torch.zeros(*self.grid_samples)
		self.b = torch.zeros(*self.grid_samples)