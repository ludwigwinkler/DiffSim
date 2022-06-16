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

from DiffEq.pytorch_CFD.scripts.Meshes import Mesh2D
from DiffEq.pytorch_CFD.scripts.Utils import LinearConvection, Diffusion, NonLinearConvection
from DiffEq.pytorch_CFD.scripts.Derivatives import finitedifference, LaplaceDerivative, PoissonDerivative
from DiffEq.pytorch_CFD.scripts.BoundaryConditions import LaplaceBoundaryCondition, PoissonBoundaryCondition

class LaplaceSolver:

	def __init__(self, mesh, forces=None):
		'''

		:param mesh: discretized u
		:param forces: list of forces that act on the mesh like convection, diffusion etc
		'''
		self.mesh = mesh
		self.dt = 0.0001

		self.derivatives = [LaplaceDerivative()]
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

class PoissonSolver:

	def __init__(self, mesh, forces=None):
		'''

		:param mesh: discretized u
		:param forces: list of forces that act on the mesh like convection, diffusion etc
		'''
		self.mesh = mesh

		self.derivatives = [PoissonDerivative()]
		self.bc = [PoissonBoundaryCondition()]

	def solve(self, t):

		num_plots = 10
		cmap = plt.cm.get_cmap('Spectral')
		colors = cmap(np.linspace(0, 1, num_plots)).tolist()  # [c1, c2, c3, ... c11]
		plot_every = t // len(colors)
		for step in range(t):
			if step % plot_every == 0:
				self.mesh.plot3d(show=True)
			for derivative in self.derivatives:
				if isinstance(derivative, NonLinearConvection):
					self.mesh = derivative(self.mesh, convect=torch.stack([self.mesh.u, self.mesh.u], dim=0), dt=self.dt)
				else:
					self.mesh = derivative(self.mesh)

			for bc in self.bc:
				bc(self.mesh)
			plt.show()

if False:
	mesh = ConvectDiffMesh()
	mesh.plot3d(show=True)
	exit()
	cfd = CFD(mesh)
	cfd.solve(t=1000)

if False:
	laplacemesh = LaplaceMesh()
	laplacemesh.plot3d(show=True)
	laplace = LaplaceSolver(laplacemesh)
	laplace.solve(t=20000)

if True:
	poissonmesh = PoissonMesh()
	poissonmesh.plot3d(show=True)
	poisson = PoissonSolver(poissonmesh)
	poisson.solve(t=100)

