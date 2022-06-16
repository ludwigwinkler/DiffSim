import future, sys, os, datetime, argparse, copy
# print(os.path.dirname(sys.executable))
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

import torch
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

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

from DiffEq.pytorch_CFD.scripts.Utils import finitedifference, LinearConvection, Diffusion, NonLinearConvection


class Mesh2D:

	def __init__(self):

		self.x_limits = [[-1,2 ], [-1.5, 0.5]] # first dim vertical, second dim horizontal
		self.grid_samples = [301, 201]
		self.dx = [(self.x_limits[0][1] - self.x_limits[0][0]) / (self.grid_samples[0] - 1),
			   (self.x_limits[1][1] - self.x_limits[1][0]) / (self.grid_samples[1] - 1)]
		self.x = [torch.linspace(start=self.x_limits[0][0], end=self.x_limits[0][1], steps=self.grid_samples[0]),
			  torch.linspace(start=self.x_limits[1][0], end=self.x_limits[1][1], steps=self.grid_samples[1])]
		self.u = torch.zeros(*self.grid_samples)
		index = lambda val, dim: int((val - self.x_limits[dim][0])/(self.x_limits[dim][1] - self.x_limits[dim][0])*self.grid_samples[dim])
		print(f'{index(0.,0)=} {index(1,0)=}')
		print(f'{index(-1,1)=} {index(0,1)=}')
		# exit()
		self.u[	index(0, 0): index(1, 0),
			index(-1, 1):index(0,1)] = 3

	def plot(self, show=True, color='red'):
		# fig = plt.figure()
		# ax = fig.gca(projection='3d')
		X, Y = np.meshgrid(mesh.x[1], mesh.x[0])
		contour = plt.contourf(X, Y, self.u.numpy(), cmap=plt.cm.get_cmap('viridis'))
		cbar = plt.colorbar(contour)
		plt.grid()
		plt.xlabel('dim2', fontsize=20)
		plt.ylabel('dim1', fontsize=20)
		if show: plt.show()


class CFD:

	def __init__(self, mesh, forces=None):
		'''

		:param mesh: discretized u
		:param forces: list of forces that act on the mesh like convection, diffusion etc
		'''
		self.mesh = mesh
		self.dt = 0.001
		# self.forces = [LinearConvection(c=[0.5, 1])]
		# self.forces = [LinearConvection(c=[0.2, 0.1]), Diffusion(nu=[.01, .01])]
		self.forces = [NonLinearConvection(c=[0.2, 0.1]), Diffusion(nu=[.01, .01])]

		# self.forces = [Diffusion(nu=.3), LinearConvection(c=3.)]
		# self.forces = [Diffusion(nu=.3), NonLinearConvection()]
		# self.forces = [NonLinearConvection()]

	def solve(self, t):

		self.num_plots = 10
		cmap = plt.cm.get_cmap('Spectral')
		colors = cmap(np.linspace(0, 1, self.num_plots)).tolist()  # [c1, c2, c3, ... c11]
		plot_every = t // len(colors)
		for step in range(t):
			if step % plot_every == 0:
				self.mesh.plot(show=False, color=colors.pop(0))
			for force in self.forces:
				if isinstance(force, NonLinearConvection):
					self.mesh = force(self.mesh, convect=torch.stack([self.mesh.u, self.mesh.u], dim=0), dt=self.dt)
				else:
					self.mesh = force(self.mesh, dt=self.dt)
			plt.show()

mesh = Mesh2D()
# mesh.plot(show=True)
cfd = CFD(mesh)
cfd.solve(t=5000)

