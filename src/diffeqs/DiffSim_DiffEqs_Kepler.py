import math
import numbers
from typing import Dict

import einops
import torch
from matplotlib import pyplot as plt

# from DiffSim.src.DiffSim_DataModules import DoublePendulum_DataModule
from DiffSim.src.diffeqs.DiffSim_DiffEqs import DiffEq, Tensor


class KeplerDiffEq(torch.nn.Module):
	def __init__(self, a, e, i, omega, Omega, mean_motion):
		super().__init__()
		self.a = a
		self.e = e
		self.i = i
		self.omega = omega
		self.Omega = Omega
		self.mean_motion = mean_motion
	
	def forward(self, mean_anomaly, x):
		mu = 3
		'''
		mean anomaly is the average speed that we need to do a full period
		it's basically n = 2π/T
		x: [orbit, [x, y, z, dx, dy, dz]]
		'''
		
		mean_anomaly = torch.ones_like(self.e).fill_(mean_anomaly)
		
		'''Computing the eccentric anomaly with Keplers Equation, can be done only numerically'''
		E = mean_anomaly.clone()
		delta_E = mean_anomaly.clone()
		step = 0
		delta_Es = []
		while abs(delta_E).mean() > 1e-10 and step < 2000:
			delta_E = (E - self.e * torch.sin(E) - mean_anomaly) / (1 - self.e * torch.cos(E))
			# print(f"\t {delta_E=}")
			delta_Es += [delta_E]
			E = E - 0.01 * delta_E
			step += 1
		
		# if 1< mean_anomaly < 1.3:
		# 	print(f"{delta_Es=}")
		# 	plt.plot(delta_Es)
		# 	plt.show()
		
		'''Computing the true anomaly'''
		true_anomaly = 2 * torch.arctan2(torch.sqrt(1 + self.e) * torch.sin(E / 2),
		                                 torch.sqrt(1 - self.e) * torch.cos(E / 2))
		
		'''Computing the heliocentric/bodycentric distance'''
		r_central = self.a * (1 - self.e * torch.cos(E))
		scaling = torch.sqrt(mu * self.a) / r_central
		
		'''Analytical position from true anomaly, BUT NOT DIFFEQ solution'''
		x_analytical = r_central * torch.cos(true_anomaly)
		y_analytical = r_central * torch.sin(true_anomaly)
		z_analytical = torch.Tensor([0.])
		
		'''Using the true position as in solving a differential equation'''
		
		assert x.dim()==2 and x.shape[-1]==6
		x_ = x[:,0].reshape(4,1)
		y_ = x[:,1].reshape(4, 1)
		z_ = x[:,2].reshape(4, 1)
		'''Velocity is given by Kepler orbit'''
		dx = - torch.sin(E) * scaling
		dy = torch.sqrt(1 - self.e ** 2) * torch.cos(E) * scaling
		dz = 0.
		
		# else:
		#
		# 	x = a * (torch.cos(E) - e)
		# 	y = a * (1 - e ** 2) ** .5 * torch.sin(E)
		# 	z = torch.Tensor([0.])
		#
		# 	dx = - torch.sin(E)  # * scaling
		# 	dy = torch.sqrt(1 - e ** 2) * torch.cos(E)  # * scaling
		# 	dz = 0
		
		'''Doing a bit of projection'''
		cosw = self.omega.cos()
		sinw = self.omega.sin()
		cosW = self.Omega.cos()
		sinW = self.Omega.sin()
		cosi = self.i.cos()
		sini = self.i.sin()
		
		r_x = (cosw * cosW - sinw * sinW * cosi) * x_ + (-sinw * cosW - cosw * sinW * cosi) * y_
		r_y = (cosw * sinW + sinw * cosW * cosi) * x_ + (-sinw * sinW + cosw * cosW * cosi) * y_
		r_z = (sinw * sini) * x_ + cosw * sini * y_
		
		dr_x = (cosw * cosW - sinw * sinW * cosi) * dx + (-sinw * cosW - cosw * sinW * cosi) * dy
		dr_y = (cosw * sinW + sinw * cosW * cosi) * dx + (-sinw * sinW + cosw * cosW * cosi) * dy
		dr_z = (sinw * sini) * dx + (cosw * sini) * dy
		
		scaling = - self.mean_motion ** 2 * self.a ** 3 / r_central ** 2
		# unit_vector_scaling = (r_x ** 2 + r_y ** 2) ** 0.5
		
		# ddr_x = scaling * r_x / unit_vector_scaling
		# ddr_y = scaling * r_y / unit_vector_scaling
		# ddr_z = torch.zeros_like(ddr_y)
		
		r = torch.concat([r_x, r_y, r_z], dim=-1)
		dr = torch.concat([dr_x, dr_y, dr_z], dim=-1)
		'''Acceleration is given by Kepler orbit by pointing to center'''
		ddr = scaling * r / r.pow(2).sum(-1, keepdim=True).pow(0.5)
		
		out = torch.concat([dr, ddr], dim=-1)
		assert out.shape==x.shape, f'{out.shape=} vs {x.shape=}'
		return out
