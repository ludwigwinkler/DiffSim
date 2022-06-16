import sys, warnings
import numbers, math
from numbers import Number
from typing import Union
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torchdyn.models
from matplotlib import cm

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch, pytorch_lightning
from torch.nn import functional as F
from torch.nn import Sequential, Module, Flatten, Unflatten, UpsamplingBilinear2d
from torch.nn import Linear, LSTM, Conv2d, LazyLinear
from torch.nn import LeakyReLU
from torchdyn.models import NeuralODE

# from torchtyping import TensorType, patch_typeguard
# from typeguard import typechecked
# patch_typeguard()

Tensor = torch.FloatTensor

# sys.path.append("/".join(os.getcwd().split("/")[:-1])) # experiments -> MLMD
# sys.path.append("/".join(os.getcwd().split("/")[:-2])) # experiments -> MLMD -> PhD

file_path = Path(__file__).absolute()
cwd = file_path.parent
phd_path = file_path
for _ in range(len(cwd.parts)):
	phd_path = phd_path.parent
	if phd_path.parts[-1] == "PhD":
		break

sys.path.append(phd_path)

from DiffSim.src.DiffSim_Utils import str2bool, warning
from DiffSim.src.DiffSim_DataModules import DoublePendulum_DataModule
from DiffSim.src.DiffSim_DiffEqs import *  # importing all the DiffEqs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FFNN(Module):
	"""
	Separate Class for Feed Forward Neural Networks for which we can set the integration direction in or before the forward method
	"""
	
	def __init__(self, in_features=None, out_features=None, hparams=None):
		
		super().__init__()
		self.hparams = hparams
		
		if out_features is None:
			out_features = in_features
		
		num_hidden = in_features * hparams.num_hidden_multiplier
		
		bias = True
		
		self.net = Sequential()
		self.net.add_module(name="Layer_0", module=Linear(in_features, num_hidden, bias=bias))
		self.net.add_module(name="Activ_0", module=LeakyReLU())
		
		for layer in range(hparams.num_layers):
			self.net.add_module(name=f"Layer_{layer + 1}", module=Linear(num_hidden, num_hidden, bias=bias), )
			self.net.add_module(name=f"Activ_{layer + 1}", module=LeakyReLU())
		
		self.net.add_module(name="Output", module=Linear(num_hidden, out_features, bias=bias))
		
		self.integration_direction = 1
	
	def forward(self, x):
		if type(self.integration_direction) is numbers.Number:
			assert self.integration_direction in [1, -1]
		if type(self.integration_direction) is torch.Tensor:
			assert self.integration_direction.shape == torch.Size([x.shape[0], 1, 1])
		
		BS, F = x.shape
		
		dx = self.integration_direction * self.net(x)
		
		return dx
	time_dependent = False

class PositionalEncoding(torch.nn.Module):
	"""
	https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
	We want to encode the absolute position t
	"""
	
	def __init__(self, t_embedding: int, max_len: int = 5000):
		super().__init__()
		
		self.t_embedding = t_embedding
		self.max_len = max_len
		"""
		div_term is used to determine the different frequencies: quickly decreasing from [1, 0.15, ..., 0.006]
		self.div_term = torch.exp(torch.arange(0, t_embedding, 2) * (-math.log(10000.0) / t_embedding))
		"""
		div_term = torch.exp(torch.arange(0, t_embedding, 2) * (-math.log(10000.0) / t_embedding))
		"""
		Our div term uses a positiv math.log(10000.0), such that the frequencies increase
		"""
		# div_term = torch.exp(torch.arange(0, t_embedding, 2) * (-math.log(1000)) / t_embedding)
		
		# position = torch.arange(max_len).unsqueeze(1) # [T,1]: [0,1,2,3,4, ... , T]
		# pe = torch.zeros(max_len, t_embedding) # [T, d_emb]
		# pe[:, 0::2] = torch.sin(position * self.div_term)
		# pe[:, 1::2] = torch.cos(position * self.div_term)
		# self.register_buffer('pe', pe)
		self.register_buffer("div_term", div_term)  # self.xyz = torch.nn.Parameter(torch.randn(100))
		
		self.plot_embeddings()
	
	def plot_embeddings(self, t=None):
		
		if t is None:
			steps = self.max_len
			t = torch.linspace(0, steps, steps).unsqueeze(-1)
			t_plot = torch.linspace(0, steps, steps).unsqueeze(-1)
		
		x = torch.randn_like(t)
		
		if t.numel() == 1:
			"""same t over entire batch"""
			t = t.unsqueeze(0).unsqueeze(0) if t.dim() == 0 else t
			t = t.repeat(x.shape[0], self.t_embedding)
			t_emb = torch.zeros(x.shape[0], self.t_embedding)  # [T, d_emb]
			t_emb[:, 0::2] = torch.sin(t[:, 0::2] * self.div_term)
			t_emb[:, 1::2] = torch.cos(t[:, 1::2] * self.div_term)
		elif t.shape == torch.Size((x.shape[0], 1)):
			t = t.repeat(1, self.t_embedding)  # [BS, 1] -> [BS, t_emb]
			t_emb = torch.zeros(x.shape[0], self.t_embedding)  # [T, t_emb]
			t_emb[:, 0::2] = torch.sin(t[:, 0::2] * self.div_term)
			t_emb[:, 1::2] = torch.cos(t[:, 1::2] * self.div_term)
		else:
			raise ValueError(f"t has wrong shape {t.shape}")
		
		assert t_emb.shape == torch.Size((x.shape[0], self.t_embedding))
		
		fig = plt.figure()
		for i in range(t_emb.shape[1]):
			plt.plot(t_plot, t_emb[:, i], label=f"{i}'th Emb")
		
		plt.close()
	
	# wandb.log({"Samples P0": [wandb.Image(fig, caption="Samples P0")]})
	
	def forward(self, t: Union[Number, Tensor], x: Tensor) -> Tensor:
		"""
		Args:
			x: Tensor, shape [seq_len, batch_size, embedding_dim]
			t: Tensor ∈ [ 0, sde_T ]
		"""
		"""
		Scaling t up to full length
		t = t * self.max_len is a bad idea
		"""
		# print(f"{self.xyz.device=}")
		assert (t.device == x.device == self.div_term.device), f"{t.device=} vs {x.device=} vs {self.div_term.device=}"
		
		if t.numel() == 1:
			"""same t over entire batch"""
			t = t.unsqueeze(0).unsqueeze(0) if t.dim() == 0 else t
			t = t.repeat(x.shape[0], self.t_embedding)
			t_emb = torch.zeros(x.shape[0], self.t_embedding, device=x.device)  # [T, d_emb]
			t_emb[:, 0::2] = torch.sin(t[:, 0::2] * self.div_term)
			t_emb[:, 1::2] = torch.cos(t[:, 1::2] * self.div_term)
		elif t.shape == torch.Size((x.shape[0], 1)):
			t = t.repeat(1, self.t_embedding)  # [BS, 1] -> [BS, t_emb]
			t_emb = torch.zeros(x.shape[0], self.t_embedding, device=x.device)  # [T, t_emb]
			t_emb[:, 0::2] = torch.sin(t[:, 0::2] * self.div_term)
			t_emb[:, 1::2] = torch.cos(t[:, 1::2] * self.div_term)
		else:
			raise ValueError(f"t has wrong shape {t.shape}")
		
		assert t_emb.shape == torch.Size((x.shape[0], self.t_embedding))
		
		return t_emb


class SkipLinear(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.linear = torch.nn.Linear(*args, **kwargs)
		self.reset_parameters()
	
	def reset_parameters(self) -> None:
		# Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
		# uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
		# https://github.com/pytorch/pytorch/issues/57109
		torch.nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
		if self.linear.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
			bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
			torch.nn.init.uniform_(self.linear.bias, -bound, bound)
	
	# self.linear.weight.data *= 0.1
	
	def forward(self, input: Tensor):
		if self.linear.in_features == self.linear.out_features:
			return input + self.linear(input)
		else:
			return self.linear(input)


class FFNN_TimeDependent(torch.nn.Module):
	time_dependent = True
	
	def __init__(self, hparams: dict = {}, direction: str = 'forward'):
		super().__init__()
		
		assert direction in ['forward', 'backward']
		self.direction = direction
		self.hparams = hparams
		self.predicting = False
		
		"""Activation functions need to be twice differentiable OpperGangnamStyle criterion"""
		actfunc = [torch.nn.Tanh, torch.nn.Sigmoid, torch.nn.ReLU][2]
		linlayer = [torch.nn.Linear, SkipLinear][1]
		dims = math.prod(hparams.data_shape)
		in_dims = dims
		
		self.x_embedding = Sequential(linlayer(in_dims, self.hparams.num_hidden),
		                              actfunc(),
		                              linlayer(self.hparams.num_hidden, self.hparams.num_hidden))
		
		self.t_embedding = PositionalEncoding(t_embedding=self.hparams.t_emb, max_len=self.hparams.data_timesteps)
		self.t_to_hidden = Sequential(Linear(self.hparams.t_emb, self.hparams.num_hidden),
		                              actfunc(),
		                              Linear(self.hparams.num_hidden, self.hparams.num_hidden))
		
		layers = []
		for l in range(self.hparams.num_layers):
			layers += [linlayer(self.hparams.num_hidden, self.hparams.num_hidden), actfunc(), ]
		
		self.nn = Sequential(*layers, Linear(self.hparams.num_hidden, dims, bias=False))
		
		# if self.hparams.data_normalization:
		# 	self.register_buffer('data_mu', torch.zeros((1, dims)))
		# 	self.register_buffer('data_std', torch.ones((1, dims)))
		
		# if self.hparams.target_normalization:
		# 	self.register_buffer('target_std', torch.ones((1, dims)))
	
	def forward(self, t: Union[torch.Tensor, Number] = None, x: torch.Tensor = None):
		'''
		We define the vector field as forward in time such that if direction=='forward', the differential is positive, whereas is 'backward' it's negative
		:param t:
		:param x:
		:return:
		'''
		
		assert x.shape[1:] == self.hparams.data_shape, f"{x.shape=} vs {self.hparams.data_shape=}"
		if self.predicting:
			assert not self.training, f"{self.predicting=} but {self.training=}"
		assert t.numel() == 1 or (t.dim() == 1 and t.shape[0] == x.shape[0]) or (
		 t.dim() == 2 and t.shape == (x.shape[0], 1)), f"{t.shape=} {x.shape=}"
		
		x_flat = torch.flatten(x, start_dim=1)
		
		t_embedding = self.t_to_hidden(self.t_embedding(t=t, x=x_flat))
		x_embedding = self.x_embedding(x_flat)
		assert x_embedding.shape == t_embedding.shape
		xt_together = x_embedding + t_embedding
		out = self.nn(xt_together)
		
		out = out.unflatten(1, self.hparams.data_shape)
		assert out.shape == x.shape, f"{out.shape=} {x.shape=}"
		return out


class Encoder(Module):
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
		actfunc = torch.nn.Tanh
		output_dim = self.hparams.latent_sim_dim
		if self.hparams.augment_latent_space:
			output_dim += self.hparams.latent_augment_dim
		bias = False
		self.nn = Sequential(Flatten(1, -1),
		                     Linear(hparams.in_features, 200, bias=bias),
		                     actfunc(),
		                     Linear(200, 100, bias=bias),
		                     actfunc(),
		                     Linear(100, 50, bias=bias),
		                     actfunc(),
		                     Linear(50, 10, bias=bias),
		                     actfunc(),
		                     Linear(10, output_dim, bias=bias), )
	
	def forward(self, x: Union[torch.Tensor, dict]):
		out = self.nn(x)
		if self.hparams.augment_latent_space:
			# assert x.shape[-1]== self.hparams.latent_sim_dim + self.hparams.latent_augment_dim, f'{x.shape=}'
			sim, augment = out.split([self.hparams.latent_sim_dim, self.hparams.latent_augment_dim], dim=-1)
			return {"latent_sim": sim, "latent_augment": augment}
		else:
			return {"latent_sim": out}


class Conv_Encoder(Module):
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
		actfunc = torch.nn.Tanh
		bias = False
		self.nn = Sequential(Conv2d(in_channels=self.hparams.data_shape[0], out_channels=5, kernel_size=5, bias=bias, ),
		                     actfunc(),
		                     Conv2d(in_channels=5, out_channels=5, kernel_size=5, bias=bias),
		                     actfunc(),
		                     Flatten(-3, -1),
		                     LazyLinear(out_features=200, bias=bias),
		                     actfunc(),
		                     Linear(in_features=200,
		                            out_features=self.hparams.analytical_latent_dim + self.hparams.latent_dim,
		                            bias=bias, ), )
	
	def forward(self, x):
		assert x.dim() in [4, 5]
		if x.dim() == 5:
			assert x.shape[1] == 1, f"{x.shape=}"
			x = x.flatten(0, 1)
		analytical_latent_dim, latent_dim = self.nn(x).split(split_size=[self.hparams.analytical_latent_dim,
		                                                                 self.hparams.latent_dim], dim=-1, )
		analytical_latent_dim = analytical_latent_dim.unsqueeze(1)
		latent_dim = latent_dim.unsqueeze(1)
		
		return {"analytical": {"analytical_mu": analytical_latent_dim}, "latent": latent_dim, }


class Conv_Decoder(Module):
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
		actfunc = torch.nn.ReLU
		bias = False
		self.nn = Sequential(Linear(self.hparams.analytical_latent_dim, 14 * 14, bias=bias),
		                     Unflatten(dim=-1, unflattened_size=(1, 14, 14)),
		                     Conv2d(in_channels=1, out_channels=5, kernel_size=5, bias=bias),
		                     actfunc(),
		                     Conv2d(in_channels=5, out_channels=1, kernel_size=5, bias=bias),
		                     UpsamplingBilinear2d(size=self.hparams.in_dims[1:]), )
	
	def forward(self, x):
		if x.dim() == 3:
			BS, T, F = x.shape
		if x.dim() == 2:
			BS, F = x.shape
		out = self.nn(x.flatten(0, 1)) if x.dim() == 3 else self.nn(x)
		
		if x.dim() == 3:
			out_shape = (BS, T, *self.hparams.out_dims)
		if x.dim() == 2:
			out_shape = (BS, *self.hparams.out_dims)
		out = out.reshape(out_shape)
		return out


class AnalyticalDecoder(Module):
	def __init__(self, hparams):
		super().__init__()
		
		self.hparams = hparams
	
	def forward(self, x: dict) -> dict:
		assert type(x) == dict
		assert all(key in ["latent_sim", "latent_augment"] for key in x.keys())
		# if len(x['latent_sim'])==2:
		# assert all(key in ['analytical_mu', 'analytical_covar'] for key in x['analytical'].keys())
		# mu, covar = x['analytical'].values()
		# assert mu.shape[-1]==self.hparams.analytical_latent_dim
		# assert covar.shape[-1]==self.hparams.latent_dim
		# assert mu.dim()==covar.dim()
		# elif len(x['analytical']) == 1:
		# 	mu = x['analytical']['analytical_mu']
		# 	covar = x['latent']
		# 	assert mu.shape[-1] == self.hparams.analytical_latent_dim
		# else:
		# 	raise ValueError(f"Wrong dictionary passed {x.keys()=} ...")
		mu = x["latent_sim"]
		covar = x["latent_augment"]
		assert mu.dim() in [3]
		if mu.dim() == 3:
			BS, T, F = mu.shape
		
		# covar=None
		
		"""
		Creating evaluation grid
		"""
		x_ = np.linspace(-15, 15, self.hparams.data_res)
		y_ = np.linspace(-15, 15, self.hparams.data_res)
		X, Y = np.meshgrid(x_, y_)
		points = Tensor(np.stack((X.flatten(), Y.flatten())).T)
		
		if mu.dim() == 3:
			"""
			covar: [2,2] -> [1,1,2,2] -> [BS,T,2,2]
			mu: [BS, T, F][:,:,0] -> [BS, T] -> [BS, T, 1, 1]
			"""
			if covar is None:
				covar = (self.hparams.data_diffusion * Tensor([[1, 0], [0, 1]]).reshape(1, 1, 2, 2).repeat(BS,
				                                                                                           T,
				                                                                                           1,
				                                                                                           1).float().contiguous())
				
				covar[..., 0, 0] = (
				 covar[..., 0, 0] + mu[..., 0] / self.hparams.data_radius * self.hparams.data_diffusion * 0.33)
				covar[..., 1, 1] = (
				 covar[..., 1, 1] + mu[..., 1] / self.hparams.data_radius * self.hparams.data_diffusion * 0.33)
				
				dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=covar, validate_args=False)
			elif type(covar) == torch.Tensor:
				# assert covar.dim()==3, f"{covar.shape=} VS {mu.shape=}"
				# assert covar.shape[-1]==self.hparams.latent_augment_dim
				
				# covar = self.hparams.data_diffusion * torch.eye(2) * covar.reshape(BS, T, 2, 2).abs()
				# covar = torch.eye(2) * torch.clamp(covar, min=0.5*self.hparams.data_diffusion)
				
				dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=covar, validate_args=False)
		else:
			raise ValueError("Something went wrong, mu and covar and not correct ...")
		
		"""
		loc: 	[BS, T, 2]
		covar:	[BS, T, 2, 2]
		points: [data_res**2, 1, 1]
		"""
		
		probs = (dist.log_prob(points.unsqueeze(1).unsqueeze(1)).exp().permute(1, 2, 0).reshape(BS,
		                                                                                        T,
		                                                                                        1,
		                                                                                        self.hparams.data_res,
		                                                                                        self.hparams.data_res))
		assert probs.dim() == 5
		min_ = (probs.flatten(-3, -1).min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
		max_ = (probs.flatten(-3, -1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
		probs = (probs - min_) / (max_ - min_)
		assert (probs.min() == 0 and probs.max() == 1.0), f"{probs.min()=} and {probs.max()=}"
		
		return probs


class HybridCircleODE(Module):
	def __init__(self, hparams={}):
		super().__init__()
		self.hparams = hparams
		self.analytical_latentdyn = AnalyticalCircleODE(hparams)
		self.latentdyn = NeuralODE(hparams)
	
	def integrate(self, x: dict, T):
		assert len(x) == 2, f"Passed arguments {x.keys()} not of length 2, bitch"
		assert all(key in ["analytical_mu", "latent"] for key in x.keys())
		
		analytical_latent, latent = x.values()
		analytical_latent_t = self.analytical_latentdyn.integrate(analytical_latent, T)
		latent_t = self.latentdyn.integrate(latent, T)
		return {"analytical": analytical_latent_t, "latent": latent_t}


class AnalyticalCircleODE(Module):
	class CircleDiffEq(Module):
		def __init__(self, hparams):
			super().__init__()
			self.hparams = hparams
			
			self.radius = torch.nn.Parameter(Tensor([self.hparams.data_radius]))
		
		def forward(self, x):
			radius = x.pow(2).sum(dim=-1, keepdim=True).pow(0.5)
			# print(f"Analytical Circle DiffEq.radius: {radius}")
			
			x_, y_ = x.chunk(chunks=2, dim=-1)
			t = torch.atan2(y_, x_)
			dx = -self.radius * torch.sin(t)
			dy = self.radius * torch.cos(t)
			
			return torch.cat([dx, dy], dim=-1)
	
	def __init__(self, hparams={}, dt=0.1):
		
		super().__init__()
		
		if len(hparams) == 0:
			warnings.warn(f"hparams is empty")
		self.hparams = hparams
		self.dt = self.hparams.data_dt
		
		ode = self.CircleDiffEq(hparams)
		self.diffeq = NeuralODE(ode, sensitivity="adjoint", solver="rk4")
	
	def integrate(self, x_init: torch.Tensor, t: Tensor = None, T: Number = None, plot=False, show=False, ):
		"""
		We need to identify the quadrant and then we can determine the correct gradient
		We use the x coordinate as the primary coordinate and the y coordinate as the secondary, case-sensitive variable

				^
		IV 	|	I
		________|__________>
		III 	|	II

		x = radius * cos(rad)
		y = radius * sin(rad)
		@param x: Tensor([x, y])
		@return:
		"""
		
		if t is not None:
			assert t.dim() == 2
			assert T is None
		elif T is not None:
			assert type(T) == Number
			assert t is None
		
		if type(x_init) == tuple:
			assert len(x_init) == 2
			x_init = x_init[0]
		else:
			assert type(x_init) == torch.Tensor
		
		if x_init is not None:
			BS, F = x_init.shape
		assert x_init.dim() == 2, f"{x_init.dim()}"
		assert x_init.shape[1] == 2, f"{x_init.shape=}"
		x, y = x_init.chunk(chunks=2, dim=-1)
		
		"""
		atan2: angle with respect to the vector (y,x)=(0,1) in [-pi, pi]
		i.e. 	torch.atan2(*Tensor([0,1])) = 0
			torch.atan2(*Tensor([1,0])) = pi/2
			torch.atan2(*Tensor([0,-1])) = pi
			torch.atan2(*Tensor([-1,0])) = -pi/2
				^
		pi/2 < θ < pi 	| 0 < θ < pi/2
		________________|__________>
		-pi < θ < - pi/2| -pi/2 < θ < 0
				|
		"""
		
		t0 = torch.atan2(y, x)
		t0 = torch.where(t0 < 0, (t0 % math.pi) + math.pi, t0)  # i.e. t0 = -0.1 -> math.pi - 0.1
		
		if t is None and T is not None:
			t = torch.linspace(start=0, end=(T + 1) * self.hparams.data_dt, steps=T + 1)
		elif t is not None and T is None:
			t = t
		else:
			exit("Integration given to AnalyticalCircleODE doesn't work ...")
		
		self.traj = self.diffeq.trajectory(x_init, t[0]).swapdims(0, 1)
		
		if plot:
			plt.plot(self.t, traj[:, 0], color="r", label="x = cos(x)")
			plt.plot(self.t, traj[:, 1], color="b", label="y = sin(x)")
			plt.legend()
			return fig
		
		if show:
			plt.show()
		
		return self.traj


class ConstantForceField(DiffEq):
	def __init__(self, hparams: dict):
		super().__init__(hparams)
		self.hparams = hparams
		self.drift = torch.nn.Parameter(torch.ones((1, self.hparams.in_features // 2)) * 0.01)
	
	@property
	def time_dependent(self):
		return False
	
	def forward(self, x, t=None):
		assert x.dim() == 2
		assert x.shape[1] == 18
		q, p = torch.chunk(x, chunks=2, dim=1)
		dx = torch.cat([torch.zeros_like(p), -self.drift * q], dim=-1)
		return dx


class DoublePendulumModel(Module):
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
		
		actfunc = [torch.nn.CELU, torch.nn.LeakyReLU, torch.nn.Tanh, torch.nn.ReLU][0]
		# self.nn = Sequential(Linear(self.hparams.in_features, 100),
		#                      actfunc(),
		#                      Linear(100, 100),
		#                      actfunc(),
		#                      Linear(50, 50), actfunc(),
		                     # Linear(100, self.hparams.out_features), )
		
		self.nn = FFNN_TimeDependent(hparams)
		
		self.nndiffeq = NNDiffEq(hparams=hparams, nn=self.nn)
		self.trainable_doublependulum_diffeq = DoublePendulumDiffEq(hparams)
		# self.trainable_doublependulum_diffeq.m1 = torch.nn.Parameter(torch.scalar_tensor(0.5))
		# self.trainable_doublependulum_diffeq.m2 = torch.nn.Parameter(torch.scalar_tensor(1.5))
		# self.trainable_doublependulum_diffeq.g = torch.nn.Parameter(torch.scalar_tensor(14.5))
		self.vectorfield = VectorField(self.trainable_doublependulum_diffeq,
		                               self.nndiffeq
		                               # DoublePendulum_SidewaysForceField(),
		                               )
		self.integrator = NeuralODE(self.vectorfield, solver=self.hparams.model_odeint, sensitivity="autograd")
	
	@staticmethod
	def model_args(parent_parser):
		parser = parent_parser.add_argument_group("DoublePendulumModel")
		parser.add_argument("--model_odeint", type=str, default="dopri5")
		parser.add_argument("--model", type=str, default="doublependulum")
		parser.add_argument("--num_hidden", type=int, default=100)
		parser.add_argument("--num_layers", type=int, default=3)
		parser.add_argument("--t_emb", type=int, default=10)
		parser.add_argument("--pretraining", type=str2bool, default=False)
		return parent_parser
	
	@torch.inference_mode()
	def viz_vector_fields(self, title=None, training_data=None):
		
		"""

		:param title:
		:param training_data: [Trajs, T, [θ_1, θ_2, dθ_1, dθ_2] ]
		:return:
		"""
		
		if training_data is not None:
			assert training_data.dim() == 3
			training_data[:, :, :2] = training_data[:, :, :2] % (2 * math.pi)
			training_data = training_data.flatten(0, 1)
		
		# x = DoublePendulum_DataModule.vectorfield_viz_input(dtheta_range=[training_data[..., 2:].min(), training_data[..., 2:].max()], theta_steps=100, dtheta_steps=100)
		x = DoublePendulum_DataModule.vectorfield_viz_input()
		histogram_range = [[x[:, 0].min(), x[:, 0].max()], [x[:, 1].min(), x[:, 1].min()], ]
		for (diffeq) in (self.vectorfield.diffeqs):  # all diffeqs operate on the same state vector space
			if diffeq.viz:
				vf = diffeq(x)
				fig, axs = plt.subplots(1, 2, sharex=True)
				axs[0].hist2d(training_data[:, 0].cpu().numpy(),
				              training_data[:, 2].cpu().numpy(),
				              bins=50,
				              density=True,
				              cmap=plt.get_cmap("OrRd"),
				              range=histogram_range, )
				axs[0].quiver(x[:, 0].cpu().numpy(),
				              x[:, 2].cpu().numpy(),
				              vf[:, 0].cpu().numpy(),
				              vf[:, 2].cpu().numpy(), )
				axs[0].axvline(math.pi)
				axs[0].axvline(0.5 * math.pi)
				axs[0].axvline(1.5 * math.pi)
				axs[0].set_xlabel(r"$\theta_1$")
				axs[0].set_ylabel(r"$d\theta_1$")
				axs[1].hist2d(training_data[:, 1].cpu().numpy(),
				              training_data[:, 3].cpu().numpy(),
				              bins=50,
				              density=True,
				              cmap=plt.get_cmap("OrRd"),
				              range=histogram_range, )
				axs[1].quiver(x[:, 1].cpu().numpy(),
				              x[:, 3].cpu().numpy(),
				              vf[:, 1].cpu().numpy(),
				              vf[:, 3].cpu().numpy(), )
				axs[1].axvline(math.pi)
				axs[1].set_xlabel(r"$\theta_2$")
				axs[1].set_ylabel(r"$d\theta_2$")
				fig.suptitle(f"{title} Model: " + str(diffeq) if title is not None else f"Model: " + str(diffeq))
				plt.tight_layout()
				plt.show()
	
	def criterion(self, pred, batch, check_args=False):
		target = batch["y"]
		assert pred.shape == batch["y"].shape, f"{pred.shape=} VS {batch['y'].shape=}"
		
		if check_args:
			if pred.dim() == 3:
				assert (torch.sum(pred[:, 0] - target[:, 0]) <= 1e-6), f" y0's aren't the same"
			elif pred.dim() == 2:
				assert torch.sum(pred[0] - target[0]) <= 1e-6, f" y0's aren't the same"
			assert pred.dim() in [3]
		
		if self.hparams.criterion == "MSE":
			loss = ((pred - target) ** 2).mean()
		elif self.hparams.criterion == "MAE":
			loss = (pred - target).abs().mean()
		elif self.hparams.criterion == "AbsE":
			loss = (((pred - target) ** 2).sum(dim=[1,
			                                        2]).mean())  # + (self.unnormalize_data(_pred).pow(2).sum(dim=-1) - 1).abs().mean()
		else:
			raise Exception(f"Wrong Criterion chosen: {self.hparams.criterion}")
		
		return loss, {}
	
	def set_output_distribution_moments(self, **kwargs):
		self.nndiffeq.set_output_distribution_moments(**kwargs)
	
	def forward(self, batch):
		# assert list(batch.keys())== ['y0', 't', 'T', 'y'], f"{batch.keys()=}"
		assert all(key in batch.keys() for key in ["y0", "t", "T"])
		t = (batch["t"] - batch["t"][:, :1])[0]  # [:1] notation maintains the dimensionality and doesnt drop the dim
		assert t.shape == (batch["t"][0].shape), f"{t.shape=} {batch['t'].shape=}"
		traj = self.integrator.trajectory(x=batch["y0"][:, 0, :], t_span=t).swapdims(0, 1)
		return traj


class ThreeBodyModel(Module):
	def __init__(self, hparams):
		super().__init__()
		self.hparams = hparams
		
		actfunc = [torch.nn.CELU, torch.nn.LeakyReLU][1]
		self.nn = Sequential(  # torch.nn.BatchNorm1d(self.hparams.in_features),
		 Linear(self.hparams.in_features, 300),
		 actfunc(),
		 Linear(300, 100),
		 actfunc(),
		 Linear(100, 50),
		 actfunc(),
		 # Linear(50, 50), actfunc(),
		 Linear(50, self.hparams.out_features), )
		
		self.nndiffeq = NNDiffEq(hparams=hparams, nn=self.nn)
		# self.diffeq = DiffEqIterator(ThreeBodyProblemDiffEq(hparams), self.nndiffeq)
		self.diffeq = VectorField(ThreeBodyProblemDiffEq(hparams),  # ThreeBodyProblem_ContractingForceField(),
		                          # ThreeBodyProblem_MLContractingForceField(),
		                          # ThreeBodyProblem_SidewaysForceField(),
		                          # self.nndiffeq
		                          # ContractingForceField(hparams)
		                          # ConstantForceField(hparams)
		                          )
		self.integrator = NeuralODE(self.diffeq, solver=self.hparams.odeint, sensitivity="adjoint")
	
	def criterion(self, pred, batch, check_args=False):
		target = batch["y"]
		assert pred.shape == batch["y"].shape, f"{pred.shape=} VS {batch['y'].shape=}"
		
		if check_args:
			if pred.dim() == 3:
				assert (torch.sum(pred[:, 0] - target[:, 0]) <= 1e-6), f" y0's aren't the same"
			elif pred.dim() == 2:
				assert torch.sum(pred[0] - target[0]) <= 1e-6, f" y0's aren't the same"
			assert pred.dim() in [3]
		
		if self.hparams.criterion == "MSE":
			loss = ((pred - target) ** 2).sum(dim=[1, 2]).mean()
		elif self.hparams.criterion == "MAE":
			loss = (pred - target).abs().mean()
		elif self.hparams.criterion == "AbsE":
			loss = (((pred - target) ** 2).sum(dim=[1,
			                                        2]).mean())  # + (self.unnormalize_data(_pred).pow(2).sum(dim=-1) - 1).abs().mean()
		else:
			raise Exception(f"Wrong Criterion chosen: {self.hparams.criterion}")
		
		return loss, {}
	
	def set_output_distribution_moments(self, **kwargs):
		if hasattr(self, "nndiffeq"):
			self.nndiffeq.set_output_distribution_moments(**kwargs)
	
	def forward(self, batch):
		# assert list(batch.keys())== ['y0', 't', 'T', 'y'], f"{batch.keys()=}"
		assert all(key in batch.keys() for key in ["y0", "t", "T"])
		t = (batch["t"] - batch["t"][:, :1])[0]  # [:1] notation maintains the dimensionality and doesnt drop the dim
		assert t.shape == (batch["t"][0].shape), f"{t.shape=} {batch['t'].shape=}"
		assert batch["y0"].dim() in [3]
		y0 = batch["y0"][:, 0, :]
		
		traj = self.integrator.trajectory(x=y0.requires_grad_(), t_span=t).swapdims(0, 1)
		return traj


class NdHamiltonianModel(Module):
	def __init__(self, hparams, analytical_vectorfield=None):
		super().__init__()
		self.hparams = hparams
		
		pytorch_lightning.utilities.seed.reset_seed()
		
		self.vectorfield = VectorField(analytical_vectorfield,
		                               # NdHamiltonianDiffEq_TimeDependent_ML_2DCircularDiffEq(hparams=hparams),
		                               NNDiffEq(hparams=hparams, nn=FFNN_TimeDependent(hparams=hparams)))
		                               # NdHamiltonianDiffEq_ML_OriginDiffEq(hparams))
		# self.vectorfield = VectorField(analytical_vectorfield,
		                               # NdHamiltonianDiffEq_TimeDependent_2DCircularDiffEq(hparams),
		                               # NdHamiltonianDiffEq_OriginDiffEq(hparams)
		                               # )
		# self.vectorfield = analytical_vectorfield
		# integrator = torchdyn.models.NeuralODE()
		self.integrator = TimeOffsetNeuralODE(vector_field=self.vectorfield, solver=self.hparams.model_odeint, sensitivity="adjoint", solver_adjoint='dopri5')
	
	@staticmethod
	def model_args(parent_parser):
		parser = parent_parser.add_argument_group("NdHamiltonianModel")
		parser.add_argument("--model_odeint", type=str, default="dopri5")
		parser.add_argument("--model", type=str, default="ndhamiltonian")
		parser.add_argument("--num_hidden", type=int, default=100)
		parser.add_argument("--num_layers", type=int, default=3)
		parser.add_argument("--t_emb", type=int, default=10)
		parser.add_argument("--pretraining", type=str2bool, default=False)
		return parent_parser
	
	def criterion(self, pred, batch, check_args=True):
		target = batch["y"]
		assert pred.shape == batch["y"].shape, f"{pred.shape=} VS {batch['y'].shape=}"
		
		if check_args:
			if pred.dim() == 3:
				assert (torch.sum(pred[:, 0] - target[:, 0]) <= 1e-6), f" y0's aren't the same"
			elif pred.dim() == 2:
				assert torch.sum(pred[0] - target[0]) <= 1e-6, f" y0's aren't the same"
			assert pred.dim() in [3]
		
		if self.hparams.criterion == "MSE":
			loss = ((pred - target) ** 2).sum(dim=[1, 2]).mean()
		elif self.hparams.criterion == "MAE":
			loss = (pred - target).abs().mean()
		elif self.hparams.criterion == "AbsE":
			loss = (((pred - target) ** 2).sum(dim=[1,
			                                        2]).mean())  # + (self.unnormalize_data(_pred).pow(2).sum(dim=-1) - 1).abs().mean()
		else:
			raise Exception(f"Wrong Criterion chosen: {self.hparams.criterion}")
		
		return loss, {}
	
	@torch.enable_grad()
	def forward(self, batch):
		# if self.vectorfield.training:
		# 	print(f"{batch['t'].shape=}")
		# 	exit(f"@ndhamiltonian.forward")
		# assert list(batch.keys())== ['y0', 't', 'T', 'y'], f"{batch.keys()=}"
		assert all(key in batch.keys() for key in ["y0", "t", "T"])
		# t = (batch["t"] - batch["t"][:, :1])[0]  # [:1] notation maintains the dimensionality and doesnt drop the dim
		# assert t.shape == (batch["t"][0].shape), f"{t.shape=} {batch['t'].shape=}"
		assert batch["y0"].dim() in [3]
		y0 = batch["y0"][:, 0, :]
		
		traj = self.integrator.trajectory(x=y0.requires_grad_(), t_span=batch['t']).swapdims(0, 1)
		
		return traj


class LatentDimLSTM(Module):
	def __init__(self, hparams):
		super().__init__()
		
		self.hparams = hparams
		
		self.lstm = LSTM(input_size=self.hparams.latent_dim, hidden_size=50, num_layers=2, batch_first=True, )
		self.out_emb = Linear(50, self.hparams.latent_dim, bias=True)
	
	def integrate(self, T, x_init):
		if x_init.dim() == 2:
			x_init = x_init.unsqueeze(1)
		assert x_init.dim() == 3
		assert t.dim() == 2
		BS, T_in, F = x_init.shape
		
		pred, (h, c) = self.lstm(x_init)
		dx = self.out_emb(pred)
		
		"""
		Training: 	[x0, x0+dx0, 	x0+dx0+dx1, 	x0+dx0+dx1+dx2 	| Autoregressive Prediction ]
		Validation:	[x0, x1, 	x2, 		x3		| Autoregressive Prediction ] = just loading up the hidden states
		"""
		out = torch.cat([x_init, x_init[:, -1:] + dx[:, -1:]], dim=1)
		for step in range(T - 1):  # because we add the first entry y0 at the beginning
			pred_t, (h, c) = self.lstm(out[:, -1:], (h, c))
			dx_t = self.out_emb(pred_t)
			out = torch.cat([out, out[:, -1:] + dx_t], dim=1)
		
		assert out.shape == (BS, T_in + T, F), f"{out.shape=} VS {(BS, T_in + T, F)}"
		return out


class LatentODE(Module):
	def __init__(self, hparams):
		
		super().__init__()
		
		self.hparams = hparams
		
		self.enc = Encoder(hparams=hparams)
		self.latent_sim_dyn = NeuralODE(CircleDiffEq(hparams))
		self.latent_augment_dyn = NeuralODE(FFNN(in_features=hparams.latent_augment_dim, hparams=hparams))
		self.dec = AnalyticalDecoder(hparams=hparams)
	
	@staticmethod
	def model_args(parent_parser):
		parser = parent_parser.add_argument_group("LatentODE")
		
		parser.add_argument("--num_hidden_multiplier", type=int, default=10)
		parser.add_argument("--num_layers", type=float, default=3)
		parser.add_argument("--odeint", type=str, default="rk4")
		parser.add_argument("--augment_latent_space", type=str2bool, default=True)
		parser.add_argument("--latent_augment_dim", type=int, default=4)
		return parent_parser
	
	def criterion(self, _pred, _target, forecasting=True, check_args=True):
		
		assert _pred.shape == _target.shape, f" {_pred.shape=} VS {_target.shape=}"
		# assert _pred.dim() == _target.dim() == 2 or _pred.dim() == _target.dim() == 3, f"{_pred.shape=}, {_target.shape=} != [Num_Traj, T, F]"
		
		if check_args:
			if _pred.dim() == 3:
				assert (torch.sum(_pred[:, 0] - _target[:, 0]) <= 1e-6), f" y0's aren't the same"
			elif _pred.dim() == 2:
				assert (torch.sum(_pred[0] - _target[0]) <= 1e-6), f" y0's aren't the same"
			
			assert _pred.dim() in [4, 5]
		
		if self.hparams.criterion == "MSE":
			return (((_pred - _target) ** 2).sum(dim=[2, 3, 4] if _pred.dim() == 5 else [2, 3]).mean())
		elif self.hparams.criterion == "MAE":
			return (_pred - _target).abs().mean()
		elif self.hparams.criterion == "AbsE":
			return (((_pred - _target) ** 2).sum(dim=[2, 3,
			                                          4]).mean())  # + (self.unnormalize_data(_pred).pow(2).sum(dim=-1) - 1).abs().mean()
		else:
			raise Exception(f"Wrong Criterion chosen: {self.hparams.criterion}")
	
	def pretrain_training_step(self, batch):
		latent = self.enc(batch["y0"])
		if not self.hparams.augment_latent_space:
			pred = latent["latent_sim"]
			loss = F.l1_loss(pred.flatten(1), batch["mu"][:, 0].flatten(1), reduction="mean")
		else:
			loss = F.l1_loss(latent["latent_sim"].flatten(1),
			                 batch["mu"][:, 0].flatten(1),
			                 reduction="mean", )  # batch['mu']:[BS, t, 1, 2] -> [BS, t=0, 2]
			covar = latent["latent_augment"].reshape_as(batch["covar"][:, 0])
			covar = torch.einsum("...ij, ...jk -> ...ik", covar, covar.swapdims(-1, -2))
			loss = loss + F.l1_loss(covar, batch["covar"][:, 0], reduction="mean")
		return {"loss": loss}
	
	def training_step(self, batch):
		pred = self(batch)
		loss = (F.l1_loss(pred[:, -1], batch["y"][:, -1], reduction="sum") / batch["y"].shape[0])
		return {"loss": loss, self.hparams.criterion: loss.detach(), "T": batch["T"]}
	
	def forward(self, batch, latent_data=None):
		
		x = batch["y0"]
		t = batch["T"]
		
		assert x.dim() == 5, f"{x.dim()=} should be [BS, t=1, C, H, W]"
		assert x.shape[1] == 1
		if type(t) == torch.Tensor:
			assert t.dim() == 2
		BS, _, C, H, W = x.shape
		
		t = torch.linspace(start=0, end=(t + 1) * self.hparams.data_dt, steps=t + 1)
		
		self.latent_t0 = self.enc(x)
		self.latent_sim_t = self.latent_sim_dyn.trajectory(x=self.latent_t0["latent_sim"], t_span=t).swapdims(0, 1)
		self.latent_augment_t = self.latent_augment_dyn.trajectory(x=self.latent_t0["latent_augment"],
		                                                           t_span=t).swapdims(0, 1)
		assert self.latent_sim_t.shape[:2] == self.latent_augment_t.shape[:2]
		
		"""
		Process covariance matrix as it has to be positive definite
		"""
		self.latent_augment_t = self.latent_augment_t.reshape_as(
		 batch["covar"].squeeze(2))  # covar[BS, T, 1, 2, 2] -> [BS, T, 2, 2]
		covar = torch.einsum("...ij, ...jk -> ...ik", self.latent_augment_t, self.latent_augment_t.swapdims(-1, -2), )
		covar = covar + 0.1 * torch.eye(2)
		
		self.out = self.dec({"latent_sim": self.latent_sim_t, "latent_augment": covar})
		
		return self.out


if __name__ == "__main__":
	
	if 1:
		circleode = CircleODE(dt=0.1)
		
		x_init = torch.stack([3 * Tensor([np.cos(rad), np.sin(rad)]) for rad in
		                      np.arange(start=-0.1, stop=0.1 + math.pi, step=0.75)])
		circleode.integrate(x_init=x_init, T=20, rad=None, plot=False, show=False)
		# print(f"{trajs.shape=} {ts.shape=}")
		# exit()
		
		fig = plt.figure(figsize=(10, 5))
		colors = [cm.jet(x) for x in np.linspace(0, 1, circleode.t.shape[0])]
		for i, (t, traj, color) in enumerate(zip(circleode.t[:10], circleode.traj[:10], colors)):
			plt.plot(t, traj[:, 0] + 0.0 * i, color=color, ls="--", label="x = cos(x)")
			plt.plot(t, traj[:, 1] + 0.0 * i, color=color, ls="-", label="y = sin(x)")
		plt.grid()
		# plt.legend()
		for pi_ in [math.pi * i for i in range(-1, 3)]:
			plt.vlines(pi_, ymin=-1, ymax=1)
		plt.show()
