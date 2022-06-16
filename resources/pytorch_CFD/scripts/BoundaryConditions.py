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


class PoissonBoundaryCondition:

	def __init__(self):
		pass

	def __call__(self, mesh):
		mesh.u[0, :] = 0
		mesh.u[-1, :] = 0
		mesh.u[:, 0] = 0
		mesh.u[:, -1] = 0

class LaplaceBoundaryCondition:

	def __init__(self):
		pass

	def __call__(self, mesh):
		'''
		p= 0 @ x=0
		p = y at x=-1
		dp/dy = 0 at y=0 & y=-1
		'''
		mesh.u[:, 0] = 0
		mesh.u[:, -1] = mesh.x[0]
		mesh.u[0, :] = mesh.u[1, :]
		mesh.u[-1, :] = mesh.u[-2, :]
		return mesh

class NavierStokesBoundaryCondition:

	def __init__(self):
		pass

	def __call__(self, mesh):
		'''
		1) dp/dx=0 at x=2 for all y
		2) dp/dy=0 at y=0 for all x
		3) dp/dx=0 at x=0 for all y
		4) p 	=0 at y=2 for all x
		Because python is call by reference, we don't have to return the object but it's changed in place by Python
		'''
		mesh.p[:,-1] = mesh.p[:,-2] # 1)
		mesh.p[0,:] = mesh.p[1,:] # 2)
		mesh.p[:,0] = mesh.p[:,1] # 3)
		mesh.p[-1,:] = 0 # 4)