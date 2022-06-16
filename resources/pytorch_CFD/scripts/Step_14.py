import future, sys, os, datetime, argparse, copy
# print(os.path.dirname(sys.executable))
import numpy as np
import matplotlib
from tqdm import tqdm, trange
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

from DiffEq.pytorch_CFD.scripts.Meshes import Mesh2D, NavierStokesMesh
from DiffEq.pytorch_CFD.scripts.Utils import LinearConvection, Diffusion, NonLinearConvection
from DiffEq.pytorch_CFD.scripts.Derivatives import difference, gradient, LaplaceDerivative, PoissonDerivative
from DiffEq.pytorch_CFD.scripts.BoundaryConditions import LaplaceBoundaryCondition, PoissonBoundaryCondition, NavierStokesBoundaryCondition

class BoundaryCondition1:

	def __init__(self):
		pass

	def pressure(self, p):
		p[:, -1] 	= p[:, -2]  # 1)
		p[0, :] 	= p[1, :]  # 2)
		p[:, 0] 	= p[:, 1]  # 3)
		p[-1, :] = 0  # 4)
		return p

	def vectorfield(self, mesh):

		mesh.u[0, :] = 0
		mesh.u[:, 0] = 0
		mesh.u[:, -1] = 0
		mesh.u[-1, :] = 1  # set velocity on cavity lid equal to 1

		mesh.v[0, :] = 0
		mesh.v[-1, :] = 0
		mesh.v[:, 0] = 0
		mesh.v[:, -1] = 0

		return mesh.u, mesh.v

class NavierStokes:

	def __init__(self):

		self.mesh = NavierStokesMesh()
		self.bc = BoundaryCondition1()
		self.rho = 1.
		self.nu = 0.1
		self.dt = 0.0001

	def compute_poisson_b(self):
		u, v = self.mesh.u, self.mesh.v
		dt, rho = self.dt, self.rho
		dy, dx = self.mesh.dx

		b =	(rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
			+ (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))
			- ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
			- 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))
			- ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

		return b

	def solve_pressure_poisson(self):
		dy, dx = self.mesh.dx

		p = self.mesh.p.clone() # we don't clone so changing self.mesh.p implicitely

		for iter in range(50):
			# print(f"{p.shape=}")
			p[1:-1, 1:-1] = (((p[1:-1,2:] + p[1:-1,:-2]) * dy**2
					+ (p[2:, 1:-1] + p[:-2,1:-1]) * dx**2) / (2* (dx**2 + dy**2))
					- dx**2 * dy**2 / (2 * (dx**2 + dy**2))
					* self.compute_poisson_b())

			'''
			1) dp/dx=0 at x=2: Pressure gradient on right side is zero
			2) dp/dy=0 at y=0: Pressure gradient on bottom is zero
			3) dp/dx=0 at x=0: Pressure gradient on left side is zero
			4) p 	=0 at y=2: Pressure on top is zero
			Because python is call by reference, we don't have to return the object but it's changed in place by Python
			'''
			p = self.bc.pressure(p)
		self.mesh.p = p.clone()

		return self.mesh.p

	def solve(self, t=50):

		dy, dx = self.mesh.dx
		dt, rho, nu = self.dt, self.rho, self.nu
		u, v = self.mesh.v, self.mesh.u
		for iter in trange(t):

			un = self.mesh.u#.clone()
			vn = self.mesh.v#.clone()

			p = self.solve_pressure_poisson()


			u[1:-1, 1:-1] +=	(-un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
					   	- vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
						- dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
						+ rho * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
						+ dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

			v[1:-1, 1:-1] += 	(- un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
						- vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
						- dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
						+ nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
						+ dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

			u, v = self.bc.vectorfield(self.mesh)

			self.mesh.u = u
			self.mesh.v = v
			self.mesh.p = p


	def plot(self):
		fig = plt.figure(figsize=(11, 7), dpi=100)

		X, Y = np.meshgrid(self.mesh.x[1], self.mesh.x[0])
		# plotting the pressure field as a contour
		plt.contourf(X, Y, self.mesh.p, alpha=0.5, cmap=plt.cm.get_cmap('viridis'))
		plt.colorbar()
		# plotting the pressure field outlines
		plt.contour(X, Y, self.mesh.p, cmap=plt.cm.get_cmap('viridis'))
		# plotting velocity field
		plt.quiver(X[::2, ::2], Y[::2, ::2], self.mesh.u[::2, ::2], self.mesh.v[::2, ::2])
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()


nv = NavierStokes()

nv.solve(t=100)
nv.plot()