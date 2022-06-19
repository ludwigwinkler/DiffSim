from typing import Union, List
from functools import partial

import matplotlib
import torch, einops
import torchdyn.models
from torch import Tensor
from torch.nn import ModuleList

matplotlib.rcParams["figure.figsize"] = [10, 10]

'''
We want a universal NeuralODE integrator that takes x0 and t_span and add t0 to all offset modules
In case t0 is actually 0, it doesnt change a thing

DiffEq should be a modified residual-style torch.nn.Module that iteratively adds every DiffEq to its input
It should therefore also replace the previous vectorfield class

So either a DiffEq implements a custom forward pass or it recursively calls child modules' forward pass

What we want as API is to just pass a bunch of DiffEqs into another DiffEq and then say: 'Accumulate all time derivatives!'
We don't want to explicitely define a forward pass because the order doesn't really matter.

DiffEq([DiffEq1, DiffEq2]):
		DiffEq1([Potential, Noise1]):
			Potential
			Noise1
		DiffEq2([Noise2]):
			Noise2
			
Recursively call DiffEq.__call__() / DiffEq.forward()

'''

torch.nn.Sequential()


class DiffEq(torch.nn.Module):
	
	def __init__(self, *diffeqs: Union[List, ModuleList, None]):
		super().__init__()
		self.diffeqs = ModuleList(diffeqs) if diffeqs is not None else None
	
	@property
	def time_dependent(self):
		'''
		Calls the attribute disguised as a property (which again calls recursively the property) of the diffeqs
		We keep recursively calling the porperty and any returns False if the list is empty
		:return:
		'''
		if self.diffeqs is not None:
			return any([diffeq_.time_dependent for diffeq_ in self.diffeqs])
	
	def forward(self, t, x, t0=0.):
		time_diff = 0
		if self.diffeqs is not None:
			for diffeq in self.diffeqs:
				if diffeq.time_dependent:
					time_diff = time_diff + diffeq(t=t, x=x, t0=t0)
				else:
					time_diff = time_diff + diffeq(t=t, x=x)
		return time_diff
	
	# def set_t0(self, t0):
	# 	if self.diffeqs is not None:
	# 		for diffeq_ in self.diffeqs:
	# 			if diffeq_.time_dependent:
	# 				diffeq_.set_t0(t0)
	
	def visualize(self, t, x):
		'''Visualizes the vector field given '''
		raise NotImplementedError


class NNDiffEq(DiffEq):
	viz = True
	
	def __init__(self, hparams, nn: torch.nn.Sequential):
		super().__init__()
		self.hparams = hparams
		self.nn = nn
	
	@property
	def time_dependent(self):
		return self.nn.time_dependent
	
	def forward(self, t, x, t0=0.):
		t = t + t0
		out = self.nn(x=x, t=t)
		return out
	
	
	def __repr__(self):
		return f"NNDiffEq"


class NeuralODE(torchdyn.models.NeuralODE):
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
		t_span can include an offset if t_span.dim()==2, then t0 = t_span[:,:1] then t0.shape=[:,1]
		We shift t_span to start with zero t_span_shifted = t_span - t_span[:,:1]
		:param x:
		:param t_span:
		:return:
		'''
		
		assert (x.shape[0] == t_span.shape[0] and t_span.dim() == 2) or t_span.dim() == 1
		if t_span.dim() == 1:
			t_span = einops.repeat(t_span, 't-> b t', b=x.shape[0])  # [T] -> [BS, T]
		t0 = t_span[:, :1]  # [BS, t0=1]
		t_span = t_span[0] - t_span[0, 0]
		assert t_span.dim() == 1 and t_span[0] == 0.
		
		old_function = self.vf.vf.forward
		self.vf.vf.forward = partial(self.vf.vf.forward, t0=t0)
		traj = super().trajectory(x=x, t_span=t_span)
		self.vf.vf.forward = old_function
		
		return traj


class Potential(DiffEq):
	def __init__(self):
		super().__init__()
	
	def forward(self, x, t):
		# print(f"Potential.forward")
		return -0.1 * x


class Noise1(DiffEq):
	time_dependent = True
	t0 = None
	
	def __init__(self):
		super().__init__()
	
	def forward(self, t, x):
		# print(f"Noise1.forward")
		print(f"{t=} {x=}")
		return 0.1 * t


class Noise2(DiffEq):
	
	def __init__(self):
		super().__init__()
	
	
	def forward(self, x, t):
		# print(f"Noise2.forward")
		return 0.1


if __name__ == "__main__":
	
	diffeq1 = DiffEq(diffeqs=[Potential(), Noise1()])
	diffeq2 = DiffEq(diffeqs=[Noise2()])
	diffeq = DiffEq(diffeqs=[diffeq1, diffeq2])
	
	for name, diffeq_ in {'diffeq': diffeq, 'diffeq1': diffeq1, 'diffeq2': diffeq2}.items():
		print(f"{name}: {diffeq_.time_dependent}")
	
	dx = diffeq(x=Tensor([2.]), t=0.)
	print(f"{dx=}\n ")
	dx1 = diffeq1(x=Tensor([2.]), t=0.)
	print(f"{dx1=} \n ")
	dx2 = diffeq2(x=Tensor([2.]), t=0.)
	print(f"{dx2=}")
	
	integrator = NeuralODE(vector_field=diffeq, solver='euler', sensitivity='adjoint', solver_adjoint='dopri5')
	t_span = torch.linspace(1, 2, 11) + 2
	print(f"{t_span=}")
	traj = integrator.trajectory(x=Tensor([2.]), t_span=t_span)
	print(f"{traj=}")
