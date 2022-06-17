from typing import Dict

import torch

from DiffSim.src.diffeqs.DiffSim_DiffEqs import DiffEq, Tensor


class ThreeBodyProblemDiffEq(DiffEq):
	
	def __init__(self, hparams: Dict, spatial_dimensions=3, num_bodies=3):
		super().__init__(hparams)
		self.spatial_dimensions = hparams.nbody_spatial_dims
		self.num_bodies = hparams.nbody_num_bodies
		
		# Define universal gravitation constant
		self.G = 6.67408e-11  # N-m2/kg2
		# Reference quantities
		self.m_nd = 1.989e+30  # kg #mass of the sun
		self.r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
		self.v_nd = 30000  # m/s #relative velocity of earth around the sun
		self.t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
		# Net constants
		self.K1 = self.G * self.t_nd * self.m_nd / (self.r_nd ** 2 * self.v_nd)
		self.K2 = self.v_nd * self.t_nd / self.r_nd
		
		# Define masses
		self.m1 = 1.1  # Alpha Centauri A
		self.m2 = 0.907  # Alpha Centauri B
		self.m3 = 1.0  # Third Star
		self.m = Tensor([self.m1, self.m2, self.m3])
	
	@property
	def time_dependent(self):
		return False
	
	def forward(self, x=None, t=None):
		'''
		ddq = dp = sum_{i!=j} G m_j / || q_j - q_i ||**3 * (q_j - q_i)
		:param x: [BS, [q, p]]
		:param t:
		:return: [BS, [dq, dp]]
		'''
		assert x.dim() == 2
		assert x.shape[1] == self.num_bodies * self.spatial_dimensions * 2, f"{x.shape=} vs {self.spatial_dimensions}*{self.num_bodies}*2"
		BS, F = x.shape
		
		'''Reshaping vector x to q=[BS, num_bodies, spatial_dims] and p=[BS, num_bodies, spatial_dims]'''
		q, p = torch.chunk(x, chunks=2, dim=-1)
		q = q.reshape(BS, self.num_bodies, self.spatial_dimensions)
		p = p.reshape(BS, self.num_bodies, self.spatial_dimensions)
		
		if False:
			r1, r2, r3 = torch.chunk(q, chunks=3, dim=1)
			v1, v2, v3 = torch.chunk(p, chunks=3, dim=1)
			# print(f"{r1.shape=}")
			r12 = torch.linalg.norm(r2 - r1)
			r13 = torch.linalg.norm(r3 - r1)
			r23 = torch.linalg.norm(r3 - r2)
			
			dv1bydt = self.K1 * self.m2 * (r2 - r1) / r12 ** 3 + self.K1 * self.m3 * (r3 - r1) / r13 ** 3
			dv2bydt = self.K1 * self.m1 * (r1 - r2) / r12 ** 3 + self.K1 * self.m3 * (r3 - r2) / r23 ** 3
			dv3bydt = self.K1 * self.m1 * (r1 - r3) / r13 ** 3 + self.K1 * self.m2 * (r2 - r3) / r23 ** 3
			dr1bydt = self.K2 * v1
			dr2bydt = self.K2 * v2
			dr3bydt = self.K2 * v3
			
			ret = torch.cat([dr1bydt, dr2bydt, dr3bydt, dv1bydt, dv2bydt, dv3bydt], dim=-1).squeeze(0)
		if True:
			
			'''
			q.shape = [BS, Bodies, 3Dim]
			q_j - q_i gives us the distance matrix with the indices [i,j]
			'''
			dist = q.unsqueeze(1) - q.unsqueeze(2)  # [BS, 1, Bodies, 3Dim] - [BS, Bodies, 1, 3Dim] = [BS, Bodies, Bodies, 3Dim]
			assert dist.shape == (BS, self.num_bodies, self.num_bodies, self.spatial_dimensions)
			dist_norm = torch.linalg.norm(dist, dim=-1)
			
			'''
			Scalar * [1, 1, 3] * [BS, Bodies=3, Bodies=3]
			The second dim of dist and dist_norm is the contribution of every other body to a specific body
			'''
			force = self.K1 * self.m.reshape((1, 1, self.num_bodies)) / (
			 dist_norm + torch.diag(torch.ones(q.shape[1]))) ** 3  # = [BS, Bodies=3, Bodies=3]
			# assert torch.allclose(force[0, 0, 1],self.K1 * self.m2 / r12**3)
			# assert torch.allclose(force[0, 0, 2],self.K1 * self.m3 / r13**3)
			
			'''
			force.shape=[BS, Bodies, Bodies], dist=[BS, Bodies, Bodies, 3Dim]
			'''
			directed_force = force.unsqueeze(-1) * dist  # [BS, Bodies, Bodies, 1] * [BS, Bodies, Bodies, 3Dim]
			directed_force = directed_force.sum(dim=-2).flatten(-2, -1)  # sum contributions of other bodies [BS, Bodies, 3Dim]
			dp = directed_force
			dq = self.K2 * p.flatten(-2, -1)
			ret = torch.cat([dq, dp],
			                dim=-1)  # assert batched_ret.shape==ret.shape, f"{batched_ret.shape=} vs {ret.shape=}"  # assert torch.allclose(batched_ret, ret)
		
		assert ret.shape == x.shape, f"{ret.shape=} and {x.shape=}"
		return ret


class ThreeBodyProblem_ContractingForceField(DiffEq):
	
	def __init__(self):
		super().__init__(hparams={})
	
	@property
	def time_dependent(self): return False
	
	def forward(self, x, t):
		assert x.dim() == 2
		# assert x.shape[1]==18
		q, p = torch.chunk(x, chunks=2, dim=-1)
		
		dp = -1.0 * q
		ret = torch.cat([torch.zeros_like(q), dp], dim=-1)
		return ret


class ThreeBodyProblem_MLContractingForceField(DiffEq):
	
	def __init__(self):
		super().__init__(hparams={})
		self.param = torch.nn.Parameter(torch.Tensor([[-1.5]]))
	
	@property
	def time_dependent(self): return False
	
	def forward(self, x, t):
		assert x.dim() == 2
		# assert x.shape[1]==18
		q, p = torch.chunk(x, chunks=2, dim=-1)
		
		dp = self.param * q
		ret = torch.cat([torch.zeros_like(q), dp], dim=-1)
		return ret


class ThreeBodyProblem_SidewaysForceField(DiffEq):
	viz = False
	
	def __init__(self):
		super().__init__(hparams={})
	
	@property
	def time_dependent(self): return False
	
	def forward(self, x, t):
		assert x.dim() == 2
		assert x.shape[1] == 18
		q, p = torch.chunk(x, chunks=2, dim=1)
		
		ret = torch.cat([torch.zeros_like(q), torch.zeros_like(q) * 0.1], dim=1)
		return ret