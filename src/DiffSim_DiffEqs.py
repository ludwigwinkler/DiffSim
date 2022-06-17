import math
import numbers
import sys
from numbers import Number
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import torchdyn.models

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch, einops
from torch.distributions import MultivariateNormal

from DiffSim.src import DiffSim_DataModules  # import datamodules directly results in circular import
from DiffSim.src.DiffSim_Utils import warning

Tensor = torch.FloatTensor

file_path = Path(__file__).absolute()
cwd = file_path.parent
phd_path = file_path
for _ in range(len(cwd.parts)):
	phd_path = phd_path.parent
	if phd_path.parts[-1] == "PhD":
		break

sys.path.append(phd_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiffEq(torch.nn.Module):
	
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
	
	@property
	def time_dependent(self):
		raise NotImplementedError
	
	def forward(self, x=None, t=None):
		raise NotImplementedError
	
	def visualize(self, x):
		'''Visualizes the vector field given '''
		raise NotImplementedError


class NNDiffEq(DiffEq):
	viz = True
	
	def __init__(self, hparams, nn: torch.nn.Sequential):
		super().__init__(hparams)
		# super(DiffEq).__init__(hparams)
		self.hparams = hparams
		self.y_mean = None
		self.y_std = None
		self.dy_mean = None
		self.dy_std = None
		
		self.nn = nn
	
	@property
	def time_dependent(self):
		return self.nn.time_dependent
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		
		q, _ = x.chunk(chunks=2, dim=-1)
		if self.hparams.nd != 2:
			print(f"{self.hparams.nd=}!=2 so viz doesn't make sense");
			return
		
		if x.dim() > 2:
			x = x.flatten(0, -2)
		
		grad = self.nn(x=x, t=t).detach().numpy()
		x = x.detach().numpy()
		
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])
	
	def forward(self, x, t=None):
		if self.time_dependent:
			t = t.expand_as(self.t0) + self.t0
		out = self.nn(x=x, t=t)
		
		return out
	
	def __repr__(self):
		return f"NNDiffEq"


class TimeOffsetNeuralODE(torchdyn.models.NeuralODE):
	'''
	We want to squeeze a time-managing function in between that handles time offsets.
	The motivation for that is that standard torchdyn.models.NeuralODE's always integrate from t=0 to T
	But time-dependent ODE require arbitrary offsets such that we need to squeeze in the offset.
	For that we need to manipulate/wrap the trajectory call to extract the offset t0 and then add it to t during the integrator calls.
	'''
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	
	def trajectory(self, x, t_span):
		'''
		t_span can include an offset if t[BS, T]
		:param x:
		:param t_span:
		:return:
		'''
		
		assert (x.shape[0] == t_span.shape[0] and t_span.dim() == 2) or t_span.dim() == 1
		if t_span.dim() == 1:
			t_span = einops.repeat(t_span, 't-> b t', b=x.shape[0])  # [T] -> [BS, T]
		t0 = t_span[:, :1]  # [BS, t0]
		
		for name, module in self.named_modules():
			if isinstance(module, DiffEq) and module.time_dependent:
				module.t0 = t0
		
		t_span = t_span[0] - t_span[0, 0]
		assert t_span.dim() == 1 and t_span[0] == 0.
		
		traj = super().trajectory(x, t_span)
		
		for name, module in self.named_modules():
			if isinstance(module, DiffEq) and module.time_dependent:
				module.t0 = None
		
		return traj


class VectorField(torch.nn.Module):
	'''
	DiffEq Summation Wrapper that takes multiple diffeqs of form f_i(x, t) and combines them to f(x, t) = \sum f_i(x, t)
	'''
	
	def __init__(self, *diffeqs: list):
		
		for diffeq in diffeqs:
			''' Checking for DiffEq Base class'''
			assert isinstance(diffeq, DiffEq), f"Differential Equation does not inherit from DiffEq, but is of type {type(diffeq)}"
		super().__init__()
		
		self.diffeqs = torch.nn.ModuleList([*diffeqs])
		self.verbose = False
	
	def forward(self, t, x):
		out = 0
		for diffeq in self.diffeqs:
			out = out + diffeq(x=x, t=t)
		return out
	
	@property
	def time_dependent(self):
		return any([diffeq.time_dependent for diffeq in self.diffeqs])
	
	def visualize(self, x=None, t=None, ax=None, show=False):
		if x.dim() > 2:
			x = x.flatten(0, -2)
		if x.shape[-1] != 4:
			print(f"{x.shape=}!=2 so viz doesn't make sense");
			return
		
		if self.time_dependent:
			for name, module in self.named_modules():
				if isinstance(module, DiffEq) and module.time_dependent:
					module.t0 = torch.zeros(x.shape[0], 1)
		
		vf = self.forward(x=x, t=t).detach().numpy()
		x = x.detach().numpy()
		ax.quiver(x[:, 0], x[:, 1], vf[:, 2], vf[:, 3])

# def __repr__(self):  # diffeqs = [str(module) for module in self.modules() if type(module) is not VectorField]  # return f"DiffEqIterator: Combined DiffEqs"  # return super().__repr__()


''' 2D DiffEqs'''


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


'''Double Pendulum DiffEqs'''


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
	
	def visualize_vector_field(self):
		'''

		:param x: [θ_1, θ_2, dθ_1, dθ_2]
		:param t:
		:return:
		θ_i ∈ [0, 2 π ]
		dθ_i ∈ [-1, 1]
		'''
		x = DiffSim_DataModules.DoublePendulum_DataModule.vectorfield_viz_input(10, 12)
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


# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)  # axs[0].quiver(x1.numpy(), y1.numpy(), dx1.numpy(), dy1.numpy())  # axs[1].quiver(x2.numpy(), y2.numpy(), dx2.numpy(), dy2.numpy())  # fig.suptitle(f"DoublePendulum Sideways Force Field \n Coord Space")  # plt.show()


'''Nd Hamiltonian DiffEqs'''


class GMM(torch.nn.Module):
	"""
	p(x) = sum_c c * p(x| μ_c, σ_c)
	"""
	
	def __init__(self, hparams):
		super().__init__()
		
		self.hparams = hparams
		self.loc = torch.clamp((torch.rand(1, self.hparams.num_gaussians, self.hparams.nd) - 0.5) * 2, -1, 1)
		self.cov = (einops.repeat(torch.eye(self.hparams.nd), "i j -> b n i j", b=1, n=self.hparams.num_gaussians, ) * 0.1)
		self.dist = MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)
		
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
		log_probs = torch.logsumexp(self.dist.log_prob(data.unsqueeze(-2)),
		                            dim=-1) - torch.log(torch.scalar_tensor(self.hparams.num_gaussians))  # [BS, Nd] -> [BS, c=1, Nd] -> [BS, c=num_gaussians] -> [BS]
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
	time_dependent = False
	
	def __init__(self, hparams: Dict, potential):
		super().__init__(hparams)
		
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
		dx = torch.cat([dHdp, -dHdq], dim=-1).detach() * self.vf_scale
		
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
	t0 = None
	
	'''
	Circling like a clock force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)
		# self.scale = self.hparams.data_dt * 0.1
		
		
		# t = torch.linspace(0, self.hparams.data_timesteps, self.hparams.data_timesteps) * self.hparams.data_dt
		# t_ = t *self.time_scale
		# plt.plot(t, t_.sin().numpy())
		# plt.show()
		self.scale = 10.
	
	# print(self)
	
	@torch.enable_grad()
	def forward(self, x=None, t=None, t0=True):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		:param t0: Bool to use t0 except when it's actively turned off
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 4, f"Can only work in 2D ..."
		assert x.shape[-1] == 2 * self.hparams.nd
		
		if t0:
			assert self.t0 is not None, f"{self.t0=} can't be None"
			t = self.t0 + t
		
		q, p = x.chunk(chunks=2, dim=-1)
		x_coord = x[:, 0]
		y_coord = x[:, 1]
		t_ = torch.atan2(y_coord, x_coord).unsqueeze(1)
		# assert t_.shape == t.shape, f"{t_.shape=} vs {t.shape=}"
		try:
			dx = torch.sin(t_) * t.cos()
		except:
			print(f"{t_.shape=} {t.shape=}")
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


class NdHamiltonianDiffEq_TimeDependent_ML_2DCircularDiffEq(DiffEq):
	time_dependent = True
	
	'''
	Circling like a clock force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)
		# self.scale = self.hparams.data_dt * 0.1
		
		
		# t = torch.linspace(0, self.hparams.data_timesteps, self.hparams.data_timesteps) * self.hparams.data_dt
		# t_ = t *self.time_scale
		# plt.plot(t, t_.sin().numpy())
		# plt.show()
		self.scale = torch.nn.Parameter(torch.scalar_tensor(5.))
		self.freq = torch.nn.Parameter(torch.scalar_tensor(0.5))
	
	# print(self)
	
	@torch.enable_grad()
	def forward(self, x=None, t=None, t0=True):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		
		assert x.shape[-1] == 4, f"Can only work in 2D ..."
		assert x.shape[-1] == 2 * self.hparams.nd
		
		if t0:
			assert self.t0 is not None, f"{self.t0=} can't be None"
			t = self.t0 + t
		
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
	time_dependent = False
	
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
	time_dependent = False
	
	'''
	Circular force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)
		self.scale = 1.
	
	# print(self)
	
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
	time_dependent = False
	
	'''
	Circular force field that undulates via a time dependetn sin(t) between pointing towards origin and away from origin
	We normalize the time to t/T ∈[0,1] and multiply it with the frequency with T= dt * time_steps: t/T * π * freq
	π* t/T ∈[0,π] because torch.sin() takes radians
	'''
	
	def __init__(self, hparams: Dict):
		super().__init__(hparams)
		self.scale = torch.nn.Parameter(torch.scalar_tensor(0.5))
	
	# print(self)
	
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
		super().__init__(hparams)
		# self.scale = self.hparams.data_dt * 0.1
		self.scale = 3.
		self.freq = 3
		self.time_scale = 1 / (self.hparams.data_dt * self.hparams.data_timesteps) * 2 * math.pi * self.freq
	
	# t = torch.linspace(0, self.hparams.data_timesteps, self.hparams.data_timesteps) * self.hparams.data_dt
	# t_ = t *self.time_scale
	# plt.plot(t, t_.sin().numpy())
	# plt.show()
	
	@torch.enable_grad()
	def forward(self, x=None, t=None):
		'''
		
		:param x: Union[ x[X,Y,T,2*nd], x[BS, T, 2*nd], x[BS, 2*nd]]
		:param t: Union[ T, BS ]
		
		We take sin(t) * c * q as the force field on the velocity where c is some constant
		'''
		assert x.shape[-1] == 2 * self.hparams.nd
		if isinstance(t, Number):
			t = Tensor([t])
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
		# ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3], scale=self.scale.data.numpy() * 10)
		ax.quiver(x[:, 0], x[:, 1], grad[:, 2], grad[:, 3])


'''Double Pendulum DiffEqs'''


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


if __name__ == "__main__":
	
	contractff = ContractingForceField()
	
	print(f"Nothing to do here ...")
