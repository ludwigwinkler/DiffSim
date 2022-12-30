import math
import numbers
from typing import Dict

import einops
import torch
from matplotlib import pyplot as plt

# from DiffSim.src.DiffSim_DataModules import DoublePendulum_DataModule
from DiffSim.src.diffeqs.DiffSim_DiffEqs import DiffEq, Tensor


class LinearForceField(DiffEq):
	
	def __init__(self, hparams={}):
		super().__init__(hparams=hparams)
	
	@property
	def time_dependent(self): return False
	
	def forward(self, x, t=None):
		x_, y_ = x.chunk(chunks=2, dim=-1)
		radius = x.pow(2).sum(dim=-1, keepdim=True).pow(0.5)
		dx_ = 2 * y_.abs() * torch.where(y_ > 0, torch.ones_like(y_), - torch.ones_like(y_))
		dy_ = 2 * y_.abs() * torch.zeros_like(dx_)
		
		return torch.cat([dx_, dy_], dim=-1)
	
	def __repr__(self):
		return 'Linear Force Field'


class ContractingForceField(DiffEq):
	
	def __init__(self, hparams={}):
		super().__init__(hparams=hparams)
	
	@property
	def time_dependent(self): return False
	
	def forward(self, x, t=None):
		x_, y_ = x.chunk(chunks=2, dim=-1)
		radius = x.pow(2).sum(dim=-1, keepdim=True).pow(0.5)
		dx_ = 0.01 * -x_
		dy_ = 0.01 * -y_
		
		return torch.cat([dx_, dy_], dim=-1)
	
	def __repr__(self):
		return 'Contracting Force Field'


class TimeDependentLinearForceField(DiffEq):
	
	def __init__(self, hparams={}):
		super().__init__(hparams=hparams)
		self.T = torch.scalar_tensor(hparams.data_dt * hparams.data_timesteps)
	
	@property
	def time_dependent(self):
		return True
	
	def forward(self, x, t):
		assert type(t) == torch.Tensor or type(t) == numbers.Number
		if type(t) is torch.Tensor:
			assert t.dim() == 2 or t.numel() == 1, f"{t.shape=}"
			if t.dim() == 2:
				assert t.shape[1] == 1
		
		x_, y_ = x.chunk(chunks=2, dim=-1)
		radius = x.pow(2).sum(dim=-1, keepdim=True).pow(0.5)
		weight_t = (self.T - t) / self.T
		dx_ = weight_t * 2 * y_.abs() * torch.where(y_ > 0, torch.ones_like(y_), - torch.ones_like(y_))
		dy_ = weight_t * 2 * y_.abs() * torch.zeros_like(dx_)
		
		return torch.cat([dx_, dy_], dim=-1)
	
	def __repr__(self):
		return f'Time Dependent Linear Force Field T={self.T}'


class CircleDiffEq(DiffEq):
	
	def __init__(self, hparams={}):
		super().__init__(hparams=hparams)
		self.radius = torch.nn.Parameter(Tensor([9.]))
	
	@property
	def time_dependent(self): return False
	
	def forward(self, x):
		radius = x.pow(2).sum(dim=-1, keepdim=True).pow(0.5)
		# print(f"{x=}, {t=}")
		x_, y_ = x.chunk(chunks=2, dim=-1)
		t = torch.atan2(y_, x_)
		dx = radius * torch.sin(t)
		dy = -radius * torch.cos(t)
		
		return -torch.cat([dx, dy], dim=-1)
	
	def __repr__(self):
		return 'Circle DiffEq'


class DoublePendulumDiffEq(DiffEq):
	m1_static = 1
	m2_static = 1
	L1_static = 1
	L2_static = 1
	viz = False
	time_dependent = False
	
	def __init__(self, hparams: Dict, mask=None):
		super().__init__(hparams)
		
		self.m1 = 1
		self.m2 = 1
		self.L1 = 1
		self.L2 = 1
		self.g = 9.81
		self.mask = mask
	
	def forward(self, x=None, t=None):
		'''

		:param x: [BS, F] = [BS, [theta1, theta2, theta1_dot, theta2_dot]]
		:param t:
		:return:
		'''
		assert x.dim() == 2, f"{x.dim()=}"
		assert x.shape[1] == 4
		theta1, theta2, dtheta1, dtheta2 = x.T  # [BS, 4] -> [4, BS]
		
		c, s = torch.cos(theta1 - theta2), torch.sin(theta1 - theta2)
		
		ddtheta1 = (self.m2 * self.g * torch.sin(theta2) * c - self.m2 * s * (self.L1 * dtheta1 ** 2 * c + self.L2 * dtheta2 ** 2) - (
		 self.m1 + self.m2) * self.g * torch.sin(theta1)) / self.L1 / (self.m1 + self.m2 * s ** 2)
		ddtheta2 = ((self.m1 + self.m2) * (
		 self.L1 * dtheta1 ** 2 * s - self.g * torch.sin(theta2) + self.g * torch.sin(theta1) * c) + self.m2 * self.L2 * dtheta2 ** 2 * s * c) / self.L2 / (
		            self.m1 + self.m2 * s ** 2)
		
		dstate = torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], dim=1)
		if self.mask is not None:
			dstate = dstate * self.mask
		
		assert dstate.shape == x.shape
		return dstate
	
	@staticmethod
	def vectorfield_viz_input(theta_steps=11, dtheta_steps=12):
		theta = torch.linspace(0, 2 * math.pi, steps=theta_steps)
		theta = einops.repeat(theta, 'theta -> (repeat theta) i', i=2, repeat=dtheta_steps)
		dtheta = torch.linspace(-2, 2, steps=dtheta_steps)
		dtheta = einops.repeat(dtheta, 'dtheta -> (repeat dtheta) i', i=2, repeat=theta_steps)
		x = torch.cat([theta, dtheta], dim=-1)
		return x
	
	def __repr__(self):
		return f'DoublePendulumDiffEq'


class DoublePendulum_SidewaysForceField(DiffEq):
	viz = True
	time_dependent = False
	
	def __init__(self, hparams={}):
		super().__init__(hparams)
	
	# self.visualize_vector_field()
	
	def forward(self, x, t=None):
		'''
		L1 = DoublePendulumDiffEq.L1_static
        L2 = DoublePendulumDiffEq.L2_static

        x1 = L1 * torch.sin(theta1)
        y1 = -L1 * torch.cos(theta1)
        x2 = x1 + L2 * torch.sin(theta2)
        y2 = y1 - L2 * torch.cos(theta2)
        
		:param x: [θ_1, θ_2, dθ_1, dθ_2]
		:param t:
		:return:
		'''
		assert x.dim() == 2
		assert x.shape[1] == 4
		theta1, theta2, dtheta1, dtheta2 = x.T
		'''
		angles theta1, theta2 start at x=1 and go counter clockwise
		'''
		# dx = torch.where(x<math.pi, -x.sin()*torch.ones_like(x), x.sin()*torch.ones_like(x)) * Tensor([[0.01, 0.01, 0, 0]])
		dx = torch.where(x < math.pi, torch.zeros_like(x), -torch.ones_like(x)) * Tensor([[0.1, 0.1, 0, 0]])
		# dx = torch.stack([dtheta1, dtheta2, torch.zeros_like(theta1), torch.zeros_like(theta1)], dim=-1)
		# assert dx.shape==x.shape
		return dx
	
	def visualize_vector_field(self, x):
		'''

		:param x: [θ_1, θ_2, dθ_1, dθ_2]
		:param t:
		:return:
		θ_i ∈ [0, 2 π ]
		dθ_i ∈ [-1, 1]
		'''
		# x = DoublePendulum_DataModule.vectorfield_viz_input(10, 12)
		vf = self.forward(x, t=None)
		
		fig, axs = plt.subplots(1, 2)
		axs[0].quiver(x[:, 0].numpy(), x[:, 2].numpy(), vf[:, 0].numpy(), vf[:, 2].numpy())
		axs[0].axvline(math.pi)
		axs[0].set_xlabel(r'$\theta_1$')
		axs[0].set_ylabel(r'$d\theta_1$')
		axs[1].quiver(x[:, 1].numpy(), x[:, 3].numpy(), vf[:, 1].numpy(), vf[:, 3].numpy())
		axs[1].axvline(math.pi)
		axs[1].set_xlabel(r'$\theta_2$')
		axs[1].set_ylabel(r'$d\theta_2$')
		fig.suptitle(f"DoublePendulum Sideways Force Field \n State Space")
		plt.tight_layout()
		plt.show()