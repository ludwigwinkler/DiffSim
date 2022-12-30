import math
from numbers import Number
from typing import Dict, Union

import einops
import torch
from torch.nn import Parameter
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal

from DiffSim.src.DiffSim_Utils import warning
from DiffSim.src.diffeqs.DiffSim_DiffEqs import DiffEq, Tensor

from torch import rand
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


class GMM(torch.nn.Module):
	"""
	p(x) = sum_c c * p(x| μ_c, σ_c)
	"""
	
	def __init__(self, hparams):
		super().__init__()
		
		self.hparams = hparams
		loc = torch.clamp((torch.rand(1, self.hparams.num_gaussians, self.hparams.nd) - 0.5) * 2, -1, 1)
		cov = (einops.repeat(torch.eye(self.hparams.nd), "i j -> b n i j", b=1, n=self.hparams.num_gaussians, ) * 0.1)
		self.loc = Parameter(loc, requires_grad=False)
		self.cov = Parameter(cov, requires_grad=False)
		# self.register_buffer(name='loc', tensor=loc)
		# self.register_buffer(name='cov', tensor=cov)
		
		self.dist = MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)
		# self.dist.loc = Parameter(self.dist.loc, requires_grad=False)
		# self.dist.covariance_matrix = Parameter(self.dist.covariance_matrix, requires_grad=False)
		
		"""Testing log_prob and prob computations"""
		data = torch.randn((23, self.hparams.nd))
		if not torch.allclose(self.prob(data), self.dist.log_prob(data.unsqueeze(-2)).exp().mean(dim=-1)):
			print(f"{(self.log_prob(data).exp()-self.prob(data))=}")
	
	def forward(self, data):
		return -self.log_prob(data)
	
	def log_prob(self, data):
		"""
		LogSumExp trick used to stabilize for weird values
		We sum over the clusters and divide by the number
		:param data: [BS, Nd]
		p(x)    = log { sum_c c * p(x| μ_c, σ_c) }
						= log { sum_c 1/c * p(x| μ_c, σ_c) }
						= log { sum_c 1/c * exp { log_prob(x|μ_c, σ_c } }
						= log { 1/c sum_c exp { log_prob(x|μ_c, σ_c } }
						= log { sum_c exp { log_prob(x|μ_c, σ_c } } - log {c}
		"""
		"""data:[BS, Nd] -> data:[BS, c=1, Nd] -> log_prob(data/c):[BS, c]"""
		# print(f"{data.device=} {self.dist.loc.device=} {self.dist.covariance_matrix.device=}")
		log_probs = torch.logsumexp(self.dist.log_prob(data.unsqueeze(-2)),
		                            dim=-1) - torch.log(torch.scalar_tensor(self.hparams.num_gaussians).to(data.device))  # [BS, Nd] -> [BS, c=1, Nd] -> [BS, c=num_gaussians] -> [BS]
		# log_probs = self.dist.log_prob(data.unsqueeze(-2)).mean(dim=-1)  # [BS, Nd] -> [BS, c=1, Nd] -> [BS, c=num_gaussians] -> [BS]
		return log_probs
	
	def prob(self, data):
		assert data.dim() in [2, 3]
		# probs = self.dist.log_prob(data.unsqueeze(-2)).exp().mean(dim=-1)  # [BS, Nd] -> [BS, c=1, Nd] -> [BS, c=num_gaussians] -> [BS]
		probs = self.log_prob(data).exp()  # [BS, Nd] -> [BS, c=1, Nd] -> [BS, c=num_gaussians] -> [BS]
		return probs
	
	def sample(self, num_samples=1):
		assert type(num_samples) == tuple
		chosen_cluster = torch.randint(low=0, high=self.hparams.num_gaussians, size=num_samples)
		samples = self.dist.sample(num_samples).squeeze(1)  # [BS, 1, c=num_gaussians, nd] -> [BS, c=num_gaussians, nd]
		chosen_cluster = (chosen_cluster.unsqueeze(-1).unsqueeze(-1).expand(*num_samples, 1, self.hparams.nd))
		# print(f"{chosen_cluster.shape=}")
		samples = torch.gather(samples, dim=1, index=chosen_cluster).squeeze(1)
		# print(f"{samples.shape=}")
		assert samples.shape == num_samples + (self.hparams.nd,), f"{samples.shape=}"
		return samples


class NdHamiltonianDiffEq(DiffEq):
	
	def __init__(self, hparams: Dict, potential):
		super().__init__()
		self.hparams = hparams
		
		self.potential = potential
		self.vf_scale = 1.
		if self.vf_scale != 1.:
			warning(f" \t NdHamiltonianDiffEq Scale = {self.vf_scale}")
	
	def forward(self, x=None, t=None):
		assert x.shape[-1] == 2 * self.hparams.nd, f"{x.shape=}"
		with torch.enable_grad():
			x.requires_grad_()
			q, p = torch.chunk(x, chunks=2, dim=-1)
			q.requires_grad_(), p.requires_grad_()
			hamiltonian = self.potential(q) + p.pow(2).sum(-1) / 2
			assert hamiltonian.shape == x.shape[:-1]
			dH, = torch.autograd.grad(hamiltonian, inputs=x, grad_outputs=torch.ones_like(hamiltonian), retain_graph=True)
		dHdq, dHdp = torch.chunk(dH, chunks=2, dim=-1)
		dx = torch.cat([dHdp, -dHdq], dim=-1) * self.vf_scale
		
		assert dx.shape == x.shape, f"{dx.shape=} {x.shape=}"
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		q, _ = x.chunk(chunks=2, dim=-1)
		if self.hparams.nd != 2:
			print(f"{self.hparams.nd=}!=2 so viz doesn't make sense");
			return
		
		mesh_shape = q.shape[:2]  # 2D
		assert q.dim() == 3 and q.shape[0] == q.shape[1] and q.shape[-1] == self.hparams.nd
		
		probs = self.potential.prob(q.flatten(0, 1))
		plotting_probs = probs.reshape(*mesh_shape)
		
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		ax.contourf(q[:, :, 0], q[:, :, 1], plotting_probs, levels=25)
		
		if x.dim() > 2:
			x = x.flatten(0, -2)
		grad = self.forward(x, t).detach().numpy()
		x = x.detach().numpy()
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


class Velocity_Integrator(DiffEq):
	time_dependent = False
	
	def __init__(self):
		super().__init__(hparams={})
	
	def forward(self, x=None, t=None):
		q, p = torch.chunk(x, chunks=2, dim=-1)
		dx = torch.cat([p, torch.zeros_like(q)], dim=-1)
		assert dx.shape == x.shape, f"{dx.shape=} {x.shape=}"
		return dx


class NdHamiltonianDiffEq_TimeDependent_2DCircularDiffEq(DiffEq):
	time_dependent = True
	'''
	Circling like a clock force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__()
		self.hparams = hparams
		self.scale = 10.
	
	@torch.enable_grad()
	def forward(self, x=None, t=None, t0=0.):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		:param t0: Bool to use t0 except when it's actively turned off
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 4, f"Can only work in 2D ..."
		assert x.shape[-1] == 2 * self.hparams.nd
		
		t = t + t0
		
		q, p = x.chunk(chunks=2, dim=-1)
		x_coord = x[:, 0]
		y_coord = x[:, 1]
		t_ = torch.atan2(y_coord, x_coord).unsqueeze(1)
		# assert t_.shape == t.shape, f"{t_.shape=} vs {t.shape=}"
		
		dx = torch.sin(t_) * t.cos()
		dy = -torch.cos(t_) * t.cos()
		dp = torch.concat([dx, dy], dim=-1)
		dx = torch.cat([torch.zeros_like(dp), self.scale * (dp - q)], dim=-1).detach()
		assert dx.shape == x.shape, f"{dx.shape=} vs {x.shape=} vs {t.shape=} {self.t0.shape=}"
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		# assert x.dim()==2, f"NdHamiltonianDiffEq input x.shape={x.shape}"
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x, t, t0=False)
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])
		
class NdHamiltonianDiffEq_CircularMoving_Attractor(DiffEq):
	time_dependent = True
	'''
	Point attractor that circles around origin point and the force of which is 1/r^2
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__()
		self.hparams = hparams
		self.scale = 1.
		self.freq = 1.
	
	@typechecked
	def forward(self, x, t: Union[Number, TensorType], t0: Union[Number, TensorType["batch", 1]]=0.):
	 
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		:param t0: Bool to use t0 except when it's actively turned off
		
		We compute position of attractor through the time index t_.
		
		
		'''
		assert x.shape[-1] == 4, f"Can only work in 2D ..."
		assert x.shape[-1] == 2 * self.hparams.nd
		assert x.dim()==2
		
		t = t + t0
		
		q, p = x.chunk(chunks=2, dim=-1)
		x_coord = x[:, 0]
		y_coord = x[:, 1]
		t_ = torch.atan2(y_coord, x_coord).unsqueeze(1)
		
		if type(t)==torch.Tensor and t.dim()==0 and t.numel()==1:
			t = torch.zeros_like(x_coord).fill_(t) + t0
		# t = einops.repeat(t, 'b t -> b (i t)', i=q.shape[-1])
		
		'''Computing x, y coordinate of attractor'''
		
		x_ = torch.sin(t)
		y_ = -torch.cos(t)
		pos = torch.stack([x_, y_], dim=-1).squeeze()
		
		assert pos.shape==q.shape, f"{pos.shape=} {q.shape=}"
		
		dp = (pos - q) / (pos - q).pow(2).sum(dim=-1, keepdim=True).pow(0.5).pow(3)
		dp = dp.clamp(-10,10)
	
		
		dx = torch.cat([torch.zeros_like(dp), self.scale * dp], dim=-1).detach()
		assert dx.shape == x.shape, f"{dx.shape=} vs {x.shape=} vs {t.shape=} {t0.shape=}"
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		# assert x.dim()==2, f"NdHamiltonianDiffEq input x.shape={x.shape}"
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x, t, t0=False)
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


class NdHamiltonianDiffEq_TimeDependent_ML_2DCircularDiffEq(DiffEq):
	time_dependent = True
	
	'''
	Circling like a clock force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)

		self.scale = torch.nn.Parameter(torch.scalar_tensor(5.))
		self.freq = torch.nn.Parameter(torch.scalar_tensor(0.5))
	
	@torch.enable_grad()
	def forward(self, x=None, t=None):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		
		assert x.shape[-1] == 4, f"Can only work in 2D ..."
		assert x.shape[-1] == 2 * self.hparams.nd
		
		q, p = x.chunk(chunks=2, dim=-1)
		x_coord = x[:, 0]
		y_coord = x[:, 1]
		t_ = torch.atan2(y_coord, x_coord).unsqueeze(1)
		dx = torch.sin(t_) * (self.freq * t).cos()
		dy = -torch.cos(t_) * (self.freq * t).cos()
		dp = torch.concat([dx, dy], dim=-1)
		dx = torch.cat([torch.zeros_like(dp), self.scale * (dp - q)], dim=-1)
		assert dx.shape == x.shape
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		# assert x.dim()==2, f"NdHamiltonianDiffEq input x.shape={x.shape}"
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x, t, t0=False).detach().numpy()
		x = x.detach().numpy()
		# ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3], scale=self.scale * 20)
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


class NdHamiltonianDiffEq_2DCircularDiffEq(DiffEq):
	
	'''
	Circling like a clock force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)
		# self.scale = self.hparams.data_dt * 0.1
		self.scale = 1.
	
	
	@torch.enable_grad()
	def forward(self, x=None, t=None):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 4, f"Can only work in 2D ..."
		assert x.shape[-1] == 2 * self.hparams.nd
		q, p = x.chunk(chunks=2, dim=-1)
		x_coord = x[:, 0]
		y_coord = x[:, 1]
		t = torch.atan2(y_coord, x_coord)
		dx = self.scale * torch.sin(t)
		dy = -self.scale * torch.cos(t)
		dp = torch.stack([dx, dy], dim=-1)
		dx = torch.cat([torch.zeros_like(dp), dp - q], dim=-1).detach()
		assert dx.shape == x.shape
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		# assert x.dim()==2, f"NdHamiltonianDiffEq input x.shape={x.shape}"
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x)
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


class NdHamiltonianDiffEq_OriginDiffEq(DiffEq):
	
	'''
	Circular force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__()
		self.hparams = hparams
		self.scale = 1.
	
	@torch.enable_grad()
	def forward(self, x=None, t=None):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 2 * self.hparams.nd
		
		q = torch.chunk(x, chunks=2, dim=-1)[0]
		dx = torch.cat([torch.zeros_like(q), -q], dim=-1).detach() * self.scale
		assert dx.shape == x.shape
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		# assert x.dim()==2, f"NdHamiltonianDiffEq input x.shape={x.shape}"
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x).detach().numpy()
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


class NdHamiltonianDiffEq_ML_OriginDiffEq(DiffEq):
	
	'''
	Circular force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)
		self.scale = torch.nn.Parameter(torch.scalar_tensor(0.5))
	
	@torch.enable_grad()
	def forward(self, x=None, t=None):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 2 * self.hparams.nd
		
		q = torch.chunk(x, chunks=2, dim=-1)[0]
		dx = torch.cat([torch.zeros_like(q), -q], dim=-1) * self.scale
		assert dx.shape == x.shape
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		# assert x.dim()==2, f"NdHamiltonianDiffEq input x.shape={x.shape}"
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x).detach().numpy()
		x = x.detach().numpy()
		# ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3], scale=self.scale.data.numpy() * 10)
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


class NdHamiltonianDiffEq_TimeDependent_OriginDiffEq(DiffEq):
	time_dependent = True
	freq = 2  # ful cycles per total time T
	
	'''
	Circular force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__()
		self.hparams = hparams
		self.scale = 1.
		self.freq = 1.
		self.time_scale = 1 / (self.hparams.data_dt * self.hparams.data_timesteps) * 2 * math.pi * self.freq
	
	
	@torch.enable_grad()
	def forward(self, x=None, t=None, t0 = 0.):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 2 * self.hparams.nd
		assert type(t)==torch.Tensor
		t = t + t0
		t = t * self.time_scale
		q = torch.chunk(x, chunks=2, dim=-1)[0]
		
		if t.dim() == 0:  # evaluate all diffeqs at same point in time
			dp = q * t.sin() * self.scale
		elif t.dim() == 1 and x.shape[-2] == t.shape[0]:  # we batched the time dimension into the data tensor
			dp = q * t.sin().unsqueeze(-1) * self.scale
		elif x.dim() == 2:  # x:[BS,2*nd]
			dp = q * t.sin() * self.scale
		else:
			exit('NDHamiltonian TimeDependent OriginDiffEq shapes shapes dont work')
		dx = torch.cat([torch.zeros_like(dp), -dp], dim=-1).detach()
		assert dx.shape == x.shape
		return dx
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		if x.dim() > 2:
			x = x.flatten(0, -2)  # t = t.flatten(0, -2)
		if ax is None:
			fig, ax = plt.subplots(1, 1)
		
		grad = self.forward(x, t).detach().numpy()
		# ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3], scale=self.scale.data.numpy() * 0.1 )
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])