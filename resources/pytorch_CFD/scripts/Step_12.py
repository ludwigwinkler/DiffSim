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

from DiffEq.pytorch_CFD.scripts.Utils import Mesh2D
from DiffEq.pytorch_CFD.scripts.Utils import finitedifference, LinearConvection, Diffusion, NonLinearConvection, LaplaceIteration


class ConvectDiffMesh(Mesh2D):

	def __init__(self):

		x_limits = [[-1,3 ], [-1, 2]] # first dim vertical, second dim horizontal [y_limit, x_limit]
		grid_samples = [301, 201]
		dx = [(x_limits[0][1] - x_limits[0][0]) / (grid_samples[0] - 1),
			   (x_limits[1][1] - x_limits[1][0]) / (grid_samples[1] - 1)]
		x = [torch.linspace(start=x_limits[0][0], end=x_limits[0][1], steps=grid_samples[0]),
			  torch.linspace(start=x_limits[1][0], end=x_limits[1][1], steps=grid_samples[1])]
		'''
		x on vertical axis, y on horizontal axis -> during plotting
		'''
		u = torch.zeros(*grid_samples)
		index = lambda val, dim: int((val - x_limits[dim][0])/(x_limits[dim][1] - x_limits[dim][0])*grid_samples[dim])
		u[	index(0, 0): index(0.5, 0),
			index(0, 1):index(0.5,1)] = 3

		Mesh2D.__init__(self,x_limits, grid_samples, dx, x, u)

class LaplaceMesh(Mesh2D):

	def __init__(self):
		x_limits = [[0, 1], [0, 2]]  # first dim vertical, second dim horizontal [y_limit, x_limit]
		grid_samples = [101, 101]
		dx = [(x_limits[0][1] - x_limits[0][0]) / (grid_samples[0] - 1),
		      (x_limits[1][1] - x_limits[1][0]) / (grid_samples[1] - 1)]
		x = [torch.linspace(start=x_limits[0][0], end=x_limits[0][1], steps=grid_samples[0]),
		     torch.linspace(start=x_limits[1][0], end=x_limits[1][1], steps=grid_samples[1])]
		'''
		x on vertical axis, y on horizontal axis -> during plotting
		'''
		u = torch.zeros(*grid_samples)

		Mesh2D.__init__(self, x_limits, grid_samples, dx, x, u)
		self = LaplaceBoundaryCondition()(self)

class LaplaceBoundaryCondition:

	def __init__(self):
		pass

	def __call__(self, mesh):
		'''
		p= 0 @ x=0
		p = y at x=-1
		dp/dy = 0 at y=0 & y=-1
		'''
		mesh.u[:,0] = 0
		mesh.u[:,-1] = mesh.x[0]
		mesh.u[0,:] = mesh.u[1,:]
		mesh.u[-1,:] = mesh.u[-2,:]
		return mesh


class LaplaceSolver:

	def __init__(self, mesh, forces=None):
		'''

		:param mesh: discretized u
		:param forces: list of forces that act on the mesh like convection, diffusion etc
		'''
		self.mesh = mesh
		self.dt = 0.0001

		self.derivatives = [LaplaceIteration()]
		self.bc = [LaplaceBoundaryCondition()]

	def solve(self, t):

		linnorm=100
		progress = tqdm(total=100)
		while linnorm>1e-4:
			u_old = self.mesh.u.clone()

			self.mesh = self.derivatives[0](self.mesh)
			self.bc[0](self.mesh)

			linnorm = torch.norm(self.mesh.u - u_old, p=1)
			desc = f'Norm( Δ u): {linnorm:.5f}'
			progress.set_description(desc)
			progress.update(1)

		self.mesh.plot3d(show=True)

if False:
	mesh = ConvectDiffMesh()
	mesh.plot3d(show=True)
	exit()
	cfd = CFD(mesh)
	cfd.solve(t=1000)

if True:
	laplacemesh = LaplaceMesh()
	laplacemesh.plot3d(show=True)
	laplace = LaplaceSolver(laplacemesh)
	laplace.solve(t=20000)

