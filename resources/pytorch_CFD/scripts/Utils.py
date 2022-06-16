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


class LinearConvection:

	def __init__(self, c):
		self.c = Tensor([c]).squeeze()

	def __call__(self, mesh, dt):

		du = finitedifference(mesh)

		if mesh.u.dim()==1 and du.dim()==1:
			mesh.u -= Tensor([self.c]) * dt *du

		elif mesh.u.dim()==2 and du.dim()==3:
			# print(f"{self.c.shape=} {diff.shape}")
			c = self.c.reshape(-1, 1, 1)
			assert c.dim()==du.dim() and c.shape[0]==du.shape[0]
			mesh.u -= torch.sum(c * dt *du,dim=0)
		else:
			raise ValueError('Invalid dims @ LinearConvection ...')
		return mesh

class NonLinearConvection:

	def __init__(self, c):
		self.c = Tensor([c]).squeeze()

	def __call__(self, mesh, convect, dt):

		du = finitedifference(mesh)

		if mesh.u.dim() == 1 and du.dim() == 1:
			mesh.u -= Tensor([self.c]) * dt * du

		elif mesh.u.dim() == 2 and du.dim() == 3:
			c = self.c.reshape(-1, 1, 1)
			assert convect.dim() == du.dim() and convect.shape[0] == du.shape[0]
			mesh.u -= torch.sum(c * convect * dt * du, dim=0)
		else:
			raise ValueError('Invalid dims @ LinearConvection ...')
		return mesh

class Diffusion:

	def __init__(self, nu):
		self.nu = Tensor([nu]).squeeze()

	def __call__(self, mesh, dt):
		ddu = finitedifference(mesh, order=2)
		if mesh.u.dim()==1 and ddu.dim()==1:
			# plt.plot(Tensor([self.nu]) * dt * ddu)
			# plt.show()
			# exit('@Diffusion')
			mesh.u += Tensor([self.nu]) * dt * ddu
		elif mesh.u.dim()==2 and ddu.dim()==3:
			nu = self.nu.reshape(-1,1,1)
			assert nu.dim() == ddu.dim() and nu.shape[0] == ddu.shape[0]
			mesh.u += torch.sum(nu * dt * ddu, dim=0)
		else:
			raise ValueError('Invalid dims @ Diffusion ...')
		return mesh


if __name__=='__main__':

	class CFD:

		def __init__(self, mesh, forces=None):
			'''

			:param mesh: discretized u
			:param forces: list of forces that act on the mesh like convection, diffusion etc
			'''
			self.mesh = mesh
			self.dt = 0.001
			# self.forces = [LinearConvection(c=.25)]


			self.forces = [Diffusion(nu=.01)]
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
					# print(f"{step=}")
					mesh.plot(show=False, color=colors.pop(0))
				for force in self.forces:
					self.mesh = force(self.mesh, self.dt)

			plt.show()

	print('start')
	mesh = Mesh1D()
	# mesh.plot()
	# exit()
	cfd = CFD(mesh)
	cfd.solve(t=1000)