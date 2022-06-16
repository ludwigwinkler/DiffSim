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

from DiffEq.pytorch_CFD.scripts.Utils import CFD, finitedifference


class Mesh2D:

	def __init__(self):
		self.x_limits = [[0, 2], [0, 2]]
		self.grid_samples = [201, 201]
		self.dx = [(self.x_limits[0][1] - self.x_limits[0][0]) / (self.grid_samples[0] - 1),
			   (self.x_limits[1][1] - self.x_limits[1][0]) / (self.grid_samples[1] - 1)]
		self.x = [torch.linspace(*self.x_limits[0], steps=self.grid_samples[0]),
			  torch.linspace(*self.x_limits[1], steps=self.grid_samples[1])]
		self.u = torch.zeros(*self.grid_samples)
		self.u[int(0.5 / self.dx[0]): int(1 / self.dx[1]), int(0.5 / self.dx[0]): int(1 / self.dx[1]) ] = 3

	def plot(self, show=True, color='red'):
		fig = plt.figure()
		ax = fig.gca(projection='3d',)
		X, Y = np.meshgrid(mesh.x[0], mesh.x[1])
		ax.plot_surface(X,Y, self.u.numpy(), cmap=plt.cm.get_cmap('viridis'))
		if show: plt.show()

mesh = Mesh2D()
# mesh.plot(show=True)
print('hi')

finitedifference(mesh)

