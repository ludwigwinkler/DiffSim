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

def difference(mesh, order, stencil='two_stencil'):

	assert stencil in ['two_stencil', 'three_stencil']
	if mesh.dim() == 1:
		'''
		One-dimensional mesh
		'''
		if order == 1:
			if stencil == 'two_stencil':
				u_padded = mesh.reshape(1, 1, -1)  # [W] -> [BS=1, C_i=1, W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1, 0], mode='replicate')  # replicate on left side
				du = F.conv1d(u_padded, weight=Tensor([-1, 1]).reshape(1, 1, -1)) / mesh.dx
				du = du.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
				return du
		if order == 2:
			u_padded = mesh.reshape(1, 1, -1)  # [W] -> [BS=1, C_i=1, W]
			u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1], mode='replicate')  # replicate on left side
			ddu = F.conv1d(u_padded, weight=Tensor([1, -2, 1]).reshape(1, 1, -1))
			ddu = ddu.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
			return ddu

	if mesh.dim() == 2:
		if order == 1:
			if stencil == 'two_stencil':
				u_padded = mesh.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1, 0, 1, 0], mode='replicate')  # replicate on left side
				'''
				first dim: y axis
				[0 -1]
				[0  1]
				second dim: x axis
				[0  0]
				[-1 1]
				'''
				weight = Tensor([[[0, -1], [0, 1]], [[0, 0], [-1, 1]]]).unsqueeze(1)
				du = F.conv2d(u_padded, weight=weight)
				du = du.squeeze(0)
				assert du.dim()==3
				return du
			if stencil=='three_stencil':
				u_padded = mesh.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1, 1, 1], mode='replicate')  # replicate on left side
				'''
				first dim:
				[ 0 1 0]
				[ 0 0 0]
				[ 0 1 0]
				second dim:
				[ 0 0 0]
				[ 1 0 1]
				[ 0 0 0]
				'''
				weight = Tensor([[[0, 1, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 1], [0, 0, 0]]]).unsqueeze(
					1)  # [C_out=1, C_in=1, H, W]
				du = F.conv2d(u_padded, weight=weight)
				du = du.squeeze(0)
				assert du.dim() == 3
				return du

		if order == 2:
			u_padded = mesh.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
			u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1, 1, 1], mode='replicate')  # replicate on all four sides
			'''
			first dim:
			[ 0  1 0]
			[ 0 -2 0]
			[ 0  1 0]
			second dim:
			[ 0  0 0]
			[ 1 -2 1]
			[ 0  0 0]
			'''
			weight = Tensor([[[0, 1, 0], [0, -2, 0], [0, 1, 0]], [[0, 0, 0], [1, -2, 1], [0, 0, 0]]]).unsqueeze(
				1)  # [C_out=1, C_in=1, H, W]
			ddu = F.conv2d(u_padded, weight=weight)
			ddu = ddu.squeeze(0)
			assert ddu.dim() == 3
			return ddu

def gradient(mesh, order=1, plot=False):
	if mesh.u.dim() == 1:
		'''
		One-dimensional mesh
		'''
		if order == 1:
			mode = ['forward', 'backward'][0]
			if mode == 'forward':
				u_padded = mesh.u.reshape(1, 1, -1)  # [W] -> [BS=1, C_i=1, W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1, 0], mode='replicate')  # replicate on left side
				du = F.conv1d(u_padded, weight=Tensor([-1, 1]).reshape(1, 1, -1)) / mesh.dx
				du = du.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
				return du
		if order == 2:
			u_padded = mesh.u.reshape(1, 1, -1)  # [W] -> [BS=1, C_i=1, W]
			u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1], mode='replicate')  # replicate on left side
			ddu = F.conv1d(u_padded, weight=Tensor([1, -2, 1]).reshape(1, 1, -1)) / mesh.dx ** 2
			ddu = ddu.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
			return ddu

	if mesh.u.dim() == 2:
		if order == 1:
			mode = ['forward', 'backward'][0]
			if mode == 'forward':
				u_padded = mesh.u.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
				u_padded = torch.nn.functional.pad(u_padded, pad=[1, 0, 1, 0], mode='replicate')  # replicate on left side
				'''
				first dim: y axis
				[0 -1]
				[0  1]
				second dim: x axis
				[0  0]
				[-1 1]
				'''
				weight = Tensor([[[0, -1], [0, 1]], [[0, 0], [-1, 1]]]).unsqueeze(1)
				du = F.conv2d(u_padded, weight=weight)
				du = du.squeeze(0) / Tensor(mesh.dx).reshape(-1, 1, 1)
				if plot:
					X, Y = np.meshgrid(mesh.x[1], mesh.x[0])
					contour = plt.contourf(X, Y, du[0].numpy(), cmap=plt.cm.get_cmap('viridis'))
					cbar = plt.colorbar(contour)
					du = du.squeeze(0).squeeze(0)  # [BS=1, C_i=1, W] -> [W]
				return du
		if order == 2:
			u_padded = mesh.u.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
			u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1, 1, 1], mode='replicate')  # replicate on all four sides
			'''
			first dim:
			[ 0  1 0]
			[ 0 -2 0]
			[ 0  1 0]
			second dim:
			[ 0  0 0]
			[ 1 -2 1]
			[ 0  0 0]
			'''
			weight = Tensor([[[0, 1, 0], [0, -2, 0], [0, 1, 0]], [[0, 0, 0], [1, -2, 1], [0, 0, 0]]]).unsqueeze(
				1)  # [C_out=1, C_in=1, H, W]
			ddu = F.conv2d(u_padded, weight=weight)
			ddu = ddu.squeeze(0) / Tensor(mesh.dx).reshape(-1, 1, 1) ** 2
			assert ddu.dim() == 3
			return ddu

class LaplaceDerivative:

	def __init__(self):
		pass

	def __call__(self, mesh):
		u_padded = mesh.u.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
		u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1, 1, 1], mode='replicate')  # replicate on all four sides
		'''
		first dim:
		[ 0 1 0]
		[ 0 0 0]
		[ 0 1 0]
		second dim:
		[ 0 0 0]
		[ 1 0 1]
		[ 0 0 0]
		'''
		weight = Tensor([[[0, 1, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 1], [0, 0, 0]]]).unsqueeze(1)  # [C_out=1, C_in=1, H, W]
		p = F.conv2d(u_padded, weight=weight).squeeze(0)
		dx = Tensor(mesh.dx).squeeze().reshape(-1, 1, 1).flip(0, 1, 2)
		p = torch.sum(dx ** 2 * p, dim=0) / (2 * (dx ** 2).sum())
		mesh.u = p

		return mesh

class PoissonDerivative:

	def __init__(self):
		pass

	def __call__(self, mesh):
		u_padded = mesh.u.unsqueeze(0).unsqueeze(0)  # [H,W] -> [BS=1, C_i=1, H,W]
		u_padded = torch.nn.functional.pad(u_padded, pad=[1, 1, 1, 1], mode='replicate')  # replicate on all four sides
		'''
		first dim:
		[ 0 1 0]
		[ 0 0 0]
		[ 0 1 0]
		second dim:
		[ 0 0 0]
		[ 1 0 1]
		[ 0 0 0]
		'''
		weight = Tensor([[[0, 1, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 1], [0, 0, 0]]]).unsqueeze(1)  # [C_out=1, C_in=1, H, W]
		p = F.conv2d(u_padded, weight=weight).squeeze(0)
		dx = mesh.dx.reshape(-1, 1, 1).flip(0, 1, 2)
		p = torch.sum(dx ** 2 * p, dim=0)
		p -= torch.prod(mesh.dx ** 2) * mesh.b

		p /= (2 * (dx ** 2).sum())
		mesh.u = p

		return mesh

class NavierStokesDerivative:

	def __init__(self):
		pass
	def __call__(self):
		'''
		Two dimensional vector: [u, v]
		We require the quantities du, dv ddu and ddv
		'''