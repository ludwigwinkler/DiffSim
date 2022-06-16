import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import wandb
import imageio, imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import torchvision

# from torchtyping import TensorType, patch_typeguard
# from typeguard import typechecked
# patch_typeguard()

fontsize = 20
params = {
 "font.size"       : fontsize,
 "legend.fontsize" : fontsize,
 "xtick.labelsize" : fontsize,
 "ytick.labelsize" : fontsize,
 "axes.labelsize"  : fontsize,
 "figure.figsize"  : (20, 10),
 "text.usetex"     : True,
 "mathtext.fontset": "stix",
 "font.family"     : "STIXGeneral", }

plt.rcParams.update(params)

from tqdm import tqdm

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from torchdyn.models import NeuralODE

file_path = Path(__file__).absolute()
cwd = file_path.parent
phd_path = file_path
for _ in range(len(cwd.parts)):
	phd_path = phd_path.parent
	if phd_path.parts[-1] == "PhD":
		break

sys.path.append(phd_path)
from DiffSim.src.DiffSim_DataSets import DiffEq_TimeSeries_DataSet
from DiffSim.src.DiffSim_Utils import str2bool
from DiffSim.src.DiffSim_DiffEqs import *  # importing all the DiffEqs


file_path = os.path.dirname(os.path.abspath(__file__)) + "/MD_DataUtils.py"
cwd = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Tensor = torch.Tensor
Scalar = torch.scalar_tensor

if not torch.cuda.is_available():
	warnings.filterwarnings("ignore", ".*does not have many workers.*")


class MinMaxScaler(object):
	"""
	Transforms each channel to the range [0, 1].
	"""
	
	def __call__(self, tensor):
		dist = tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]
		dist[dist == 0.0] = 1.0
		scale = 1.0 / dist
		tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
		return tensor


def np_array(_tensor):
	if _tensor is not None:
		assert isinstance(_tensor, torch.Tensor)
		return _tensor.cpu().squeeze().detach().numpy()
	else:
		return None


class Circular_GaussianBlob_DataModule(LightningDataModule):
	def __init__(self, hparams):
		super().__init__()
		self.hparams.update(hparams.__dict__)
		
		self.data_shape = (1, hparams.data_res, hparams.data_res)
		
		self.latent_sim_dim = 2
		self.latent_augment_dim = 4
	
	@staticmethod
	def datamodule_args(parent_parser):
		parser = parent_parser.add_argument_group("Circular_GaussianBlob_DataModule")
		
		parser.add_argument("--data_res", type=int, default=29)
		parser.add_argument("--data_diffusion", type=float, default=3)
		parser.add_argument("--data_dt", type=float, default=0.05)
		parser.add_argument("--data_timesteps", type=int, default=3000)
		parser.add_argument("--data_radius", type=int, default=5)
		parser.add_argument("--batch_size", type=int, default=128)
		parser.add_argument("--train_traj_repetition", type=int, default=10000)
		parser.add_argument("--val_split", type=float, default=0.8)
		parser.add_argument("--num_workers", type=int, default=4 * torch.cuda.device_count() if torch.cuda.is_available() else 0, )
		parser.add_argument("--pretraining", type=str2bool, default=True)
		
		return parent_parser
	
	def __repr__(self):
		return f"Train: {self.data_train} \t Val: {self.data_val}"
	
	def prepare_data(self) -> None:
		"""
		np.cos works with values between 0 and 2*np.pi -> 0...6.28
		"""
		
		num_datasets = 1
		
		x = np.linspace(-15, 15, self.hparams.data_res)
		y = np.linspace(-15, 15, self.hparams.data_res)
		
		X, Y = np.meshgrid(x, y)
		points = Tensor(np.stack((X.flatten(), Y.flatten())).T)
		
		self.radius = 6
		
		rads = (np.linspace(start=0, stop=self.hparams.data_timesteps, num=self.hparams.data_timesteps, ) * self.hparams.data_dt)
		
		x = self.hparams.data_radius * np.cos(rads)
		y = self.hparams.data_radius * np.sin(rads)
		T = self.hparams.data_timesteps
		
		mus = torch.from_numpy(np.vstack([x, y])).T.reshape(-1, 1, 2).float()
		covar = (self.hparams.data_diffusion * Tensor([[1, 0], [0, 1]]).reshape(1, 1, 2, 2).repeat(T, 1, 1, 1).contiguous().float())  # [T, 1, 2, 2]
		covar = self.covar_postprocessing(mus, covar)
		
		dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=covar)
		
		probs = (dist.log_prob(points).exp().reshape(self.hparams.data_timesteps, self.hparams.data_res, self.hparams.data_res, ))
		data = probs
		
		if 0:
			for step, time_step in enumerate(data):
				if step < 20:
					plt.matshow(time_step.numpy())
					plt.title(f"{step}")
					plt.show()
		
		self.rawdata = data.unsqueeze(1).unsqueeze(0)
		self.latent_data = (mus.unsqueeze(0), covar.unsqueeze(0))
		
		time = torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps)
		self.time = einops.repeat(time, "t -> d t", d=self.rawdata.shape[0])
		
		assert (self.time.shape[:2] == self.rawdata.shape[:2] == self.latent_data[0].shape[:2] == self.latent_data[1].shape[:2])
	
	def covar_postprocessing(self, mu, covar):
		"""
		Construct LightningDataModule specific covariance matrix
		"""
		offdiag = mu[:, :, 0]
		covar[:, :, 0, 1] += (0.75 * self.hparams.data_diffusion * offdiag / offdiag.max())
		covar[:, :, 1, 0] += (0.75 * self.hparams.data_diffusion * offdiag / offdiag.max())
		return covar
	
	def setup(self, stage: Optional[str] = None) -> None:
		
		assert self.rawdata.dim() == 5, f"Should be [DataSet, Timestep, Channel, H, W]"
		assert self.rawdata.shape[1] == self.hparams.data_timesteps
		
		"""[DataSet, T, C, H, W] -> [DataSet, T, C*H*W] -> min(dim=-1) -> min = [DataSet, T, 1, 1, 1]"""
		data_min = (torch.min(self.rawdata.flatten(-3, -1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
		data_max = (torch.max(self.rawdata.flatten(-3, -1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
		self.data = (self.rawdata - data_min) / (data_max - data_min + 1e-8)
		self.data_mean = self.data.mean(dim=[0, 1])
		self.data_std = self.data.std(dim=[0, 1])
		
		train_size = int(self.data.shape[1] * self.hparams.val_split)
		val_size = self.data.shape[1] - train_size
		
		split_data = lambda data: torch.split(data, [train_size, val_size], dim=1)
		
		train_data, val_data = split_data(self.data)
		train_latent_data_mu, val_latent_data_mu = split_data(self.latent_data[0])
		train_latent_data_covar, val_latent_data_covar = split_data(self.latent_data[1])
		
		train_latent_data = (train_latent_data_mu, train_latent_data_covar)
		val_latent_data = (val_latent_data_mu, val_latent_data_covar)
		train_time, val_time = split_data(self.time)
		
		self.data_train = DiffEq_TimeSeries_DataSet(data=train_data,
		                                            time=train_time,
		                                            other_data=train_latent_data,
		                                            input_length=self.hparams.input_length,
		                                            output_length=self.hparams.output_length_train,
		                                            output_length_sampling=self.hparams.iterative_output_length_training,
		                                            traj_repetition=self.hparams.train_traj_repetition,
		                                            sample_axis="trajs", )
		self.data_val = DiffEq_TimeSeries_DataSet(data=val_data,
		                                          time=val_time,
		                                          other_data=val_latent_data,
		                                          input_length=self.hparams.input_length,
		                                          output_length=self.hparams.output_length_val,
		                                          output_length_sampling=False,
		                                          traj_repetition=1,
		                                          sample_axis="timesteps", )
	
	def collate_fn(self, batch):
		
		y0 = []
		t = []
		T = batch[0]["T"]
		y = []
		mu = []
		covar = []
		
		for sample in batch:
			y0_, t_, _, y_, latent_data_ = sample.values()
			mu_, covar_ = latent_data_
			y0 += [y0_]
			t += [t_]
			y += [y_]
			mu += [mu_]
			covar += [covar_]
		
		y0 = torch.stack(y0)
		t = torch.stack(t)
		y = torch.stack(y)
		mu = torch.stack(mu)
		covar = torch.stack(covar)
		
		return {"y0": y0, "t": t, "T": T, "y": y, "mu": mu, "covar": covar}
	
	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		
		dataloader = DataLoader(self.data_train,
		                        batch_size=self.hparams.batch_size,
		                        shuffle=True,
		                        num_workers=self.hparams.num_workers,
		                        collate_fn=self.collate_fn, )
		
		return dataloader
	
	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val,
		                  batch_size=self.hparams.batch_size * 2,
		                  num_workers=self.hparams.num_workers,
		                  shuffle=False,
		                  collate_fn=self.collate_fn, )
	
	def plot_prediction(self, pred, target, t, title="", show=False):
		
		assert pred.dim() == target.dim()
		
		if pred.dim() == target.dim() == 5:
			random_batch_idx = int(np.random.uniform() * pred.shape[0])  # int(U[0,1] * BS)
			batch_y_ = target[random_batch_idx]  # .sum(dim=0, keepdim=True)
			pred_ = pred[random_batch_idx]  # .sum(dim=0, keepdim=True)
		elif pred.dim() == 4:
			batch_y_ = target
			pred_ = pred
		fig, axs = plt.subplots()
		axs.imshow(torchvision.utils.make_grid(torch.cat([batch_y_, pred_], dim=0), nrow=t + 1, pad_value=-1).permute(1, 2, 0)[:, :, 0])
		
		axs.set_title(title)
		if show:
			plt.show()
		
		return fig
	
	def generate_gif(self, pred, target, t, title="", file_str="../experiments/giffygiffy2.gif", show=False, ):
		
		assert pred.dim() == target.dim() == 5
		random_batch_idx = int(np.random.uniform() * pred.shape[0])
		figs = []
		for pred_, target_ in zip(pred[random_batch_idx], target[random_batch_idx]):
			fig, axs = plt.subplots(1, 2)
			axs[0].imshow(pred_.permute(1, 2, 0))
			axs[1].imshow(target_.permute(1, 2, 0))
			axs[0].set_title("Prediction")
			axs[1].set_title("Target")
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			figs += [image]
			plt.close()
		
		imageio.mimsave(file_str, figs, fps=20)


class ForceField_Circular_GaussianBlob_DataModule(LightningDataModule):
	def __init__(self, hparams):
		super().__init__()
		self.hparams.update(hparams.__dict__)
		
		self.data_shape = (1, hparams.data_res, hparams.data_res)
		
		self.analytical_latent_dim = (2, 4)
	
	@torch.no_grad()
	def visualize_force_fields(self, diffeqiterator, plot=True):
		
		assert type(diffeqiterator) == VectorField
		x = np.linspace(-15, 15, self.hparams.data_res // 2)
		y = np.linspace(-15, 15, self.hparams.data_res // 2)
		X, Y = np.meshgrid(x, y)
		points = Tensor(np.stack((X.flatten(), Y.flatten())).T)
		
		for forcefield in diffeqiterator.children():
			if not forcefield.time_dependent:
				dx = forcefield(points)
				plt.quiver(points[:, 0], points[:, 1], dx[:, 0], dx[:, 1])
				plt.title(str(forcefield))
				if plot:
					plt.show()
			elif forcefield.time_dependent:
				x = np.linspace(-15, 15, self.hparams.data_res // 4)
				y = np.linspace(-15, 15, self.hparams.data_res // 4)
				X, Y = np.meshgrid(x, y)
				points = Tensor(np.stack((X.flatten(), Y.flatten())).T)
				fig, axs = plt.subplots(3, 2)
				axs = axs.flatten()
				t = torch.linspace(start=0, end=self.hparams.data_timesteps * self.hparams.data_dt, steps=6, )
				for i, t_ in enumerate(t):
					dx_t = forcefield(points, t_)
					axs[i].quiver(points[:, 0], points[:, 1], dx_t[:, 0], dx_t[:, 1], scale=50)
					axs[i].set_title(f"Time: {t_}")
				fig.suptitle(str(forcefield))
				plt.tight_layout()
			
			if plot:
				plt.show()
	
	def visualize_latent_data(self, pred_dict, true_dict, show=True):
		
		pred = pred_dict["analytical"]["analytical_mu"]
		true = true_dict["analytical"]["analytical_mu"]
		
		assert pred.shape == true.shape, f"{pred.shape=} {true.shape=}"
		
		random_batch_idx = torch.randperm(pred.shape[0])[0]
		pred = pred[random_batch_idx].T
		true = true[random_batch_idx].T
		
		plt.scatter(*pred, alpha=0.5, color="red", label="Pred")
		plt.scatter(*true, alpha=0.5, color="blue", label="True")
		
		if show:
			plt.show()
	
	def prepare_data(self) -> None:
		"""
		np.cos works with values between 0 and 2*np.pi -> 0...6.28
		"""
		
		x = np.linspace(-15, 15, self.hparams.data_res)
		y = np.linspace(-15, 15, self.hparams.data_res)
		
		X, Y = np.meshgrid(x, y)
		points = Tensor(np.stack((X.flatten(), Y.flatten())).T).float()
		
		diffeqs = VectorField(CircleDiffEq(), LinearForceField(), TimeDependentLinearForceField(self.hparams), )
		
		self.visualize_force_fields(diffeqs)
		
		t = (torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps, ) * self.hparams.data_dt)
		
		diffeqs.set_t0(t[:1].unsqueeze(0))
		diffeq = NeuralODE(diffeqs, sensitivity="adjoint", solver="rk4")
		
		with torch.no_grad():
			traj = diffeq.trajectory(Tensor([[0, self.hparams.data_radius]]), t).swapdims(0, 1)
		
		T = self.hparams.data_timesteps
		x, y = traj[0].T  # Traj.shape=[1,T,2] -> x,y=[2,T]
		
		mus = torch.from_numpy(np.vstack([x, y])).T.float().reshape(-1, 2)
		covar = (self.hparams.data_diffusion * Tensor([[1, 0], [0, 1]]).unsqueeze(0).repeat(T, 1, 1).float().contiguous())  # [T, 1, 2, 2]
		covar = self.covar_postprocessing(mus, covar)
		
		dist = torch.distributions.MultivariateNormal(mus.unsqueeze(1), covariance_matrix=covar.unsqueeze(1), validate_args=True)
		probs = (dist.log_prob(points).exp().reshape(self.hparams.data_timesteps, self.hparams.data_res, self.hparams.data_res, ))
		
		min_ = probs.flatten(-2, -1).min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
		max_ = probs.flatten(-2, -1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
		probs = (probs - min_) / (max_ - min_)
		assert (probs.min() == 0 and probs.max() == 1.0), f"{probs.min()=} and {probs.max()=}"
		data = probs
		
		mus = mus.unsqueeze(0)
		covar = covar.unsqueeze(0)
		assert mus.shape == (1, self.hparams.data_timesteps, 2)
		assert covar.shape == (1, self.hparams.data_timesteps, 2, 2)
		
		self.time = t
		self.rawdata = data.unsqueeze(1).unsqueeze(0)  # [T, H, W] -> [1, T, 1, H, W]
		self.latent_data = (mus, covar)  # [1, T, 2], [1, T, 2, 2]
	
	def covar_postprocessing(self, mu, covar):
		"""
		Construct LightningDataModule specific covariance matrix
		"""
		
		covar[..., 0, 0] += (self.hparams.data_diffusion * 0.75 * (mu[..., 0] / self.hparams.data_radius).clamp(min=-self.hparams.data_radius,
		                                                                                                        max=self.hparams.data_radius))
		covar[..., 1, 1] += (self.hparams.data_diffusion * 0.75 * (mu[..., 1] / self.hparams.data_radius).clamp(min=-self.hparams.data_radius,
		                                                                                                        max=self.hparams.data_radius))
		
		covar = torch.eye(2) * torch.clamp(covar, min=0.25 * self.hparams.data_diffusion, max=1.75 * self.hparams.data_diffusion, )
		
		assert covar[..., 0, 0].min() > 0.0, f"{covar[..., 0, 0].min()=}"
		assert covar[..., 1, 1].min() > 0.0, f"{covar[..., 1, 1].min()=}"
		
		return covar
	
	def setup(self, stage: Optional[str] = None) -> None:
		
		assert (self.rawdata.dim() == 5), f"Should be [DataSet, Timestep, Channel, H, W], but is {self.rawdata.shape=}"
		assert self.rawdata.shape[1] == self.hparams.data_timesteps
		
		"""[DataSet, T, C, H, W] -> [DataSet, T, C*H*W] -> min(dim=-1) -> min = [DataSet, T, 1, 1, 1]"""
		# data_min = torch.min(self.rawdata.flatten(-3,-1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		# data_max = torch.max(self.rawdata.flatten(-3,-1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		self.data = self.rawdata
		self.data_mean = self.data.mean(dim=[0, 1])
		self.data_std = self.data.std(dim=[0, 1])
		
		train_size = int(self.data.shape[1] * self.hparams.val_split)
		val_size = self.data.shape[1] - train_size
		
		train_data, val_data = torch.split(self.data, [train_size, val_size], dim=1)
		train_latent_data = (self.latent_data[0][:, :train_size], self.latent_data[1][:, :train_size],)
		val_latent_data = (self.latent_data[0][:, train_size:], self.latent_data[1][:, train_size:],)
		train_time = self.time[:train_size]
		val_time = self.time[train_size:]
		
		train_data = self.data
		train_latent_data = self.latent_data
		train_time = self.time
		
		self.data_train = DiffEq_TimeSeries_DataSet(data=train_data,
		                                            time=train_time,
		                                            other_data=train_latent_data,
		                                            input_length=self.hparams.input_length,
		                                            output_length=self.hparams.output_length_train,
		                                            output_length_sampling=self.hparams.output_length_sampling,
		                                            traj_repetition=self.hparams.train_traj_repetition,
		                                            sample_axis="timesteps" if self.hparams.train_traj_repetition == 1 else "trajs", )
		self.data_val = DiffEq_TimeSeries_DataSet(data=val_data,
		                                          time=val_time,
		                                          other_data=val_latent_data,
		                                          input_length=self.hparams.input_length,
		                                          output_length=self.hparams.output_length_val,
		                                          output_length_sampling=False,
		                                          traj_repetition=1,
		                                          sample_axis="timesteps", )
	
	def collate_fn(self, batch):
		
		y0 = []
		t = []
		T = batch[0][2]
		y = []
		mu = []
		covar = []
		
		for sample in batch:
			y0_, t_, T_, y_, latent_data_ = sample
			mu_, covar_ = latent_data_
			y0 += [y0_]
			t += [t_]
			y += [y_]
			mu += [mu_]
			covar += [covar_]
		
		y0 = torch.stack(y0)
		t = torch.stack(t)
		y = torch.stack(y)
		mu = torch.stack(mu)
		covar = torch.stack(covar)
		
		return {"y0": y0, "t": t, "T": T, "y": y, "mu": mu, "covar": covar}
	
	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		
		dataloader = DataLoader(self.data_train,
		                        batch_size=self.hparams.batch_size,
		                        shuffle=True,
		                        num_workers=self.hparams.num_workers,
		                        collate_fn=self.collate_fn, )
		
		return dataloader
	
	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val,
		                  batch_size=self.hparams.batch_size * 2,
		                  num_workers=self.hparams.num_workers,
		                  shuffle=False,
		                  collate_fn=self.collate_fn, )
	
	def plot_prediction(self, pred, target, pred_latent, latent, t, title="", show=False):
		
		assert pred.dim() == target.dim()
		
		if pred.dim() == target.dim() == 5:
			random_batch_idx = int(np.random.uniform() * pred.shape[0])  # int(U[0,1] * BS)
			batch_y_ = target[random_batch_idx]  # .sum(dim=0, keepdim=True)
			pred_ = pred[random_batch_idx]  # .sum(dim=0, keepdim=True)
		# elif pred.dim()==4:
		# 	batch_y_ = target.
		# 	pred_ = pred
		
		fig, axs = plt.subplots(1, 2)
		axs = axs.flatten()
		axs[0].matshow(target.permute(1, 2, 0))
		axs[0].plot(*latent.T, c="red")
		axs[0].set_title("Ground Truth")
		axs[1].imshow(pred.permute(1, 2, 0))
		axs[1].plot(*pred_latent.T, c="red")
		axs[1].set_title("Hybrid ML DiffEq")
		# axs.imshow(torchvision.utils.make_grid(torch.cat([batch_y_, pred_], dim=0), nrow=t + 1, pad_value=-1).permute(1, 2, 0)[:, :, 0])
		
		fig.suptitle(title)
		
		return fig
	
	def generate_gif(self, pred, target, t, pred_latent, latent, title="", file_str="../experiments/giffygiffy.gif", show=False, ):
		
		assert pred.dim() == target.dim() == 5
		
		random_batch_idx = int(np.random.uniform() * pred.shape[0])
		pred_latent = pred_latent[random_batch_idx] + Tensor([[14.5, 14.5]])
		latent = latent[random_batch_idx] + Tensor([[14.5, 14.5]])
		figs = []
		for pred_, target_ in zip(pred[random_batch_idx], target[random_batch_idx]):
			fig = self.plot_prediction(pred_, target_, pred_latent, latent, t, title, False)
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			figs += [image]
			plt.close(fig)
		
		imageio.mimsave(file_str, figs, fps=10)
		plt.clf()
	
	def __repr__(self):
		return f"Train: {self.data_train} \t Val: {self.data_val}"


class DoublePendulum_DataModule(LightningDataModule):
	time_dependent = False
	
	def __init__(self, hparams):
		super().__init__()
		
		"""Add own hparams to already existing hparams and check keys"""
		self.hparams.update(hparams.__dict__)
		
		self.data_shape = (4,)
		self.in_features = math.prod(self.data_shape)
		self.out_features = self.in_features
	
	# self.prepare_data()  # self.setup()  # self.viz_vector_fields()  # self.viz_prediction()
	
	@staticmethod
	def datamodule_args(parent_parser, timesteps=3000, dt=0.01, num_trajs=100, train_traj_repetition=1000, ):
		parser = parent_parser.add_argument_group("DoublePendulum_DataModule")
		
		parser.add_argument("--data_dt", type=float, default=dt)
		parser.add_argument("--data_timesteps", type=int, default=timesteps)
		parser.add_argument("--train_traj_repetition", type=int, default=train_traj_repetition)
		parser.add_argument("--val_split", type=float, default=0.8)
		# parser.add_argument("--num_workers", type=int, default=0)
		parser.add_argument("--data_odeint", type=str, default="rk4")
		parser.add_argument("--num_trajs", type=int, default=num_trajs)
		
		return parent_parser
	
	def example_trajectory(self):
		
		x0 = Tensor([[3 * np.pi / 7, 0, 3 * np.pi / 4, 0]])
		
		diffeqs = VectorField(self.doublependulum_diffeq, DoublePendulum_SidewaysForceField())
		# t = torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps) * self.hparams.data_dt
		t = torch.linspace(start=0, end=100, steps=100) * self.hparams.data_dt
		
		diffeq = NeuralODE(diffeqs, sensitivity="adjoint", solver="rk4")
		
		with torch.inference_mode():
			traj = diffeq.trajectory(x0, t).swapdims(0, 1)
		
		from matplotlib.patches import Circle
		
		assert traj.dim() == 3
		theta1, theta2, dtheta1, dtheta2 = traj[0].T
		
		L1 = self.doublependulum_diffeq.L1
		L2 = self.doublependulum_diffeq.L2
		x1 = L1 * torch.sin(theta1)
		y1 = -L1 * torch.cos(theta1)
		x2 = x1 + L2 * torch.sin(theta2)
		y2 = y1 - L2 * torch.cos(theta2)
		
		assert x1.shape == x2.shape == torch.Size([traj.shape[1]])
		r = 0.05
		trail_secs = 1
		max_trail = int(trail_secs / self.hparams.data_dt)
		
		figs = []
		for t in range(traj.shape[1])[1::5]:
			fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
			ax = fig.add_subplot(111)
			
			ax.plot([0, x1[t], x2[t]], [0, y1[t], y2[t]], lw=2, c="k")
			# Circles representing the anchor point of rod 1, and bobs 1 and 2.
			c0 = Circle((0, 0), r / 2, fc="k", zorder=10)
			c1 = Circle((x1[t], y1[t]), r, fc="b", ec="b", zorder=10)
			c2 = Circle((x2[t], y2[t]), r, fc="r", ec="r", zorder=10)
			ax.add_patch(c0)
			ax.add_patch(c1)
			ax.add_patch(c2)
			
			# The trail will be divided into ns segments and plotted as a fading line.
			ns = 20
			s = max_trail // ns
			
			for j in range(ns):
				imin = t - (ns - j) * s
				if imin < 0:
					continue
				imax = imin + s + 1
				# The fading looks better if we square the fractional length along the
				# trail.
				alpha = (j / ns) ** 2
				ax.plot(x2[imin:imax], y2[imin:imax], c="r", solid_capstyle="butt", lw=2, alpha=alpha, )
			
			# Centre the image on the fixed anchor point, and ensure the axes are equal
			ax.set_xlim(-L1 - L2 - r, L1 + L2 + r)
			ax.set_ylim(-L1 - L2 - r, L1 + L2 + r)
			ax.set_aspect("equal", adjustable="box")
			plt.axis("off")
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			plt.close(fig)
			figs += [image]
		
		assert len(figs) > 10, f"{len(figs)=}"
		imageio.mimsave("./DoublePendulumExample.gif", figs, fps=10)
	
	
	def prepare_data(self) -> None:
		"""
		θ_i = 0 -> 6 o'clock
		θ_i = 0.5 π -> 3 o'clock
		θ_i = π -> 0/12 o'clock
		:return:
		"""
		q0 = torch.zeros(self.hparams.num_trajs, 2).uniform_(0.0 * math.pi, 2 * math.pi)
		p0 = torch.zeros(self.hparams.num_trajs, 2).uniform_(-1, 1)
		x0 = torch.cat([q0, p0], dim=-1)
		
		self.vectorfield = VectorField(DoublePendulumDiffEq(hparams=self.hparams),
		                               DoublePendulum_SidewaysForceField(),
		                               # TimeDependentLinearForceField(self.hparams)
		                               )
		
		t = (torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps, ) * self.hparams.data_dt)
		
		# diffeq = TimeDependentNeuralODE(vector_field=diffeqs, sensitivity='adjoint', solver=self.hparams.data_odeint)
		diffeq = NeuralODE(vector_field=self.vectorfield, sensitivity="adjoint", solver=self.hparams.data_odeint, )
		
		with torch.inference_mode():
			traj = diffeq.trajectory(x0, t).swapdims(0, 1)
		
		self.time = t.unsqueeze(0).repeat(self.hparams.num_trajs, 1)
		self.rawdata = traj
	
	@staticmethod
	def vectorfield_viz_input(theta_range=[0, 2 * math.pi], dtheta_range=[-1, 1], theta_steps=11, dtheta_steps=12, ):
		theta = torch.linspace(0, 2 * math.pi, steps=theta_steps)
		theta = einops.repeat(theta, "theta -> (repeat theta) i", i=2, repeat=dtheta_steps)
		dtheta = torch.linspace(-1, 1, steps=dtheta_steps)
		dtheta = einops.repeat(dtheta, "dtheta -> (repeat dtheta) i", i=2, repeat=theta_steps)
		x = torch.cat([theta, dtheta], dim=-1)
		return x
	
	def setup(self, stage: Optional[str] = None) -> None:
		
		assert (self.rawdata.dim() == 3), f"Should be [DataSet, Timestep, Channel, H, W], but is {self.rawdata.shape=}"
		assert self.rawdata.shape[1] == self.hparams.data_timesteps
		assert (self.rawdata.shape[:2] == self.time.shape), f"{self.rawdata.shape=} vs {self.time.shape=}"
		
		"""[DataSet, T, F] -> [DataSet, T, F] -> min(dim=-1) -> min = [DataSet, T, 1, 1, 1]"""
		# data_min = torch.min(self.rawdata.flatten(-3,-1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		# data_max = torch.max(self.rawdata.flatten(-3,-1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		self.data = self.rawdata
		self.data_mean = self.data.mean(dim=[0, 1])
		self.data_std = self.data.std(dim=[0, 1])
		
		self.y_mean = self.data_mean
		self.y_std = self.data_std
		dy = self.data[0, 1:] - self.data[0, :-1]
		self.dy_mean = dy.mean(dim=0, keepdim=True)
		self.dy_std = dy.std(dim=0, keepdim=True)
		
		print(f"DataModule: {self.dy_mean.mean()=} {self.dy_std.mean()=}")
		
		train_size = int(self.data.shape[1] * self.hparams.val_split)
		val_size = self.data.shape[1] - train_size
		
		self.train_data, self.val_data = torch.split(self.data, [train_size, val_size], dim=1)
		self.train_time, self.val_time = torch.split(self.time, [train_size, val_size], dim=1)
		
		assert self.train_time.dim() == self.val_time.dim() == 2
		
		self.data_train = DiffEq_TimeSeries_DataSet(data=self.train_data,
		                                            time=self.train_time,
		                                            other_data=None,
		                                            input_length=self.hparams.input_length,
		                                            output_length=self.hparams.output_length_train,
		                                            output_length_increase=self.hparams.iterative_output_length_training_increase,
		                                            traj_repetition=self.hparams.train_traj_repetition,
		                                            sample_axis="timesteps" if self.hparams.train_traj_repetition <= 1 else "trajs", )
		self.data_val = DiffEq_TimeSeries_DataSet(data=self.val_data,
		                                          time=self.val_time,
		                                          other_data=None,
		                                          input_length=self.hparams.input_length,
		                                          output_length=self.hparams.output_length_val,
		                                          traj_repetition=1,
		                                          sample_axis="timesteps", )
	
	@torch.inference_mode()
	def viz_vector_fields(self, title=None):
		
		if self.train_data is not None:
			assert self.train_data.dim() == 3
			training_data = self.train_data.clone()
			training_data[:, :, :2] = training_data[:, :, :2] % (2 * math.pi)
			training_data = training_data.flatten(0, 1)
		
		# x = DoublePendulum_DataModule.vectorfield_viz_input(dtheta_range=[training_data[:,2:].min(), training_data[:, 2:].max()], theta_steps=100, dtheta_steps=100)
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
				axs[0].quiver(x[:, 0].cpu().numpy(), x[:, 2].cpu().numpy(), vf[:, 0].cpu().numpy(), vf[:, 2].cpu().numpy(), )
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
				axs[1].quiver(x[:, 1].cpu().numpy(), x[:, 3].cpu().numpy(), vf[:, 1].cpu().numpy(), vf[:, 3].cpu().numpy(), )
				axs[1].axvline(math.pi)
				axs[1].set_xlabel(r"$\theta_2$")
				axs[1].set_ylabel(r"$d\theta_2$")
				fig.suptitle(f"{title} DataModule: \n" + str(diffeq) if title is not None else f"DataModule: \n" + str(diffeq))
				plt.tight_layout()
				plt.show()
	
	def collate_fn(self, batch):
		
		y0 = []
		t = []
		T = batch[0]["T"]
		y = []
		
		for sample in batch:
			y0_, t_, T_, y_, latent_data_ = sample.values()
			
			y0 += [y0_]
			t += [t_]
			y += [y_]
		
		y0 = torch.stack(y0)
		t = torch.stack(t)
		y = torch.stack(y)
		
		return {"y0": y0, "t": t, "T": T, "y": y}
	
	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		
		dataloader = DataLoader(self.data_train,
		                        batch_size=self.hparams.batch_size,
		                        shuffle=True,
		                        num_workers=self.hparams.num_workers,
		                        collate_fn=self.collate_fn, )
		
		return dataloader
	
	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val,
		                  batch_size=self.hparams.batch_size * 2,
		                  num_workers=self.hparams.num_workers,
		                  shuffle=False,
		                  collate_fn=self.collate_fn, )
	
	@staticmethod
	def state_to_coord(state):
		assert state.dim() in [2, 3]
		assert state.shape[-1] == 4
		
		if state.dim() == 3:
			BS, T, F = state.shape
			theta1, theta2, dtheta1, dtheta2 = einops.rearrange(state, "b t f -> f b t")
			assert theta1.shape == torch.Size((BS, T))
		elif state.dim() == 2:
			T, F = state.shape
			theta1, theta2, dtheta1, dtheta2 = einops.rearrange(state, "t f -> f t")
			assert theta1.shape == torch.Size((T,))
		
		L1 = DoublePendulumDiffEq.L1_static
		L2 = DoublePendulumDiffEq.L2_static
		
		x1 = L1 * torch.sin(theta1)
		y1 = -L1 * torch.cos(theta1)
		x2 = x1 + L2 * torch.sin(theta2)
		y2 = y1 - L2 * torch.cos(theta2)
		
		return x1, y1, x2, y2
	
	@torch.inference_mode()
	def viz_prediction(self, model=None, title=""):
		
		"""Create prediction data"""
		max_length = 500  # to speed up plotting in case we have super long validation trajectories
		y0 = self.val_data[:1, :1, :]  # [Traj=1, T=1, F=4]
		y = self.val_data[:1, :max_length]  # [Traj=1, T, F=4]
		t = self.val_time[:1, :max_length]  # [Traj=1, T]
		steps = t.numel()
		
		if model is not None:
			pred = model({"y0": y0, "t": t, "T": t})
			target = y
		else:
			pred = y
			target = y
		
		"""Viz prediction"""
		from matplotlib.patches import Circle
		
		assert pred.dim() == target.dim() == 3
		
		"""data[BS, T, 4]"""
		# pred_theta1, pred_theta2, pred_dtheta1, pred_dtheta2 = pred[0].T
		# target_theta1, target_theta2, target_dtheta1, target_dtheta2 = target[0].T
		BS, T, F = pred.shape
		
		pred_x1, pred_y1, pred_x2, pred_y2 = self.state_to_coord(pred[0])
		target_x1, target_y1, target_x2, target_y2 = self.state_to_coord(target[0])
		
		# plt.plot(target[0,:,0], label=r'$\theta_1$')
		# plt.plot(target[0,:,1], label=r'$\theta_2$')
		# plt.legend()
		# plt.show()
		
		L1 = DoublePendulumDiffEq.L1_static
		L2 = DoublePendulumDiffEq.L2_static
		r = 0.05
		trail_secs = 1
		max_trail = int(trail_secs / self.hparams.data_dt)
		
		overlay = True
		
		figs = []
		for t in tqdm(range(0, T, 5)):  # t is in fact the step
			
			if not overlay:
				
				fig, axs = plt.subplots(figsize=(16.3333, 6.25), dpi=72, ncols=2, nrows=1)
				axs = axs.flatten()
				
				axs[0].plot([0, target_x1[t], target_x2[t]], [0, target_y1[t], target_y2[t]], lw=2, c="k", )
				c0 = Circle((0, 0), r / 2, fc="k", zorder=10)
				c1 = Circle((target_x1[t], target_y1[t]), r, fc="b", ec="b", zorder=10)
				c2 = Circle((target_x2[t], target_y2[t]), r, fc="r", ec="r", zorder=10)
				axs[0].add_patch(c0)
				axs[0].add_patch(c1)
				axs[0].add_patch(c2)
				
				axs[1].plot([0, pred_x1[t], pred_x2[t]], [0, pred_y1[t], pred_y2[t]], lw=2, c="k", )
				# Circles representing the anchor point of rod 1, and bobs 1 and 2.
				c0 = Circle((0, 0), r / 2, fc="k", zorder=10)
				c1 = Circle((pred_x1[t], pred_y1[t]), r, fc="b", ec="b", zorder=10)
				c2 = Circle((pred_x2[t], pred_y2[t]), r, fc="r", ec="r", zorder=10)
				axs[1].add_patch(c0)
				axs[1].add_patch(c1)
				axs[1].add_patch(c2)
				
				# The trail will be divided into ns segments and plotted as a fading line.
				ns = 20
				s = max_trail // ns
				
				for j in range(ns):
					imin = t - (ns - j) * s
					if imin < 0:
						continue
					imax = imin + s + 1
					# The fading looks better if we square the fractional length along the trail.
					alpha = (j / ns) ** 2
					axs[0].plot(target_x2[imin:imax], target_y2[imin:imax], c="r", solid_capstyle="butt", lw=2, alpha=alpha, )
					axs[1].plot(pred_x2[imin:imax], pred_y2[imin:imax], c="r", solid_capstyle="butt", lw=2, alpha=alpha, )
				
				# Centre the image on the fixed anchor point, and ensure the axes are equal
				axs[0].set_xlim(-L1 - L2 - r, L1 + L2 + r)
				axs[1].set_xlim(-L1 - L2 - r, L1 + L2 + r)
				axs[0].set_ylim(-L1 - L2 - r, L1 + L2 + r)
				axs[1].set_ylim(-L1 - L2 - r, L1 + L2 + r)
				axs[0].set_aspect("equal", adjustable="box")
				axs[1].set_aspect("equal", adjustable="box")
				plt.axis("off")
			
			elif overlay:
				
				fig = plt.figure(figsize=(8.3333, 6.25), dpi=72, )
				ax = fig.add_subplot(111)
				
				ax.plot([0, target_x1[t], target_x2[t]], [0, target_y1[t], target_y2[t]], lw=2, c="k", label="Ground Truth", )
				c0 = Circle((0, 0), r / 2, fc="k", zorder=10)
				c1 = Circle((target_x1[t], target_y1[t]), r, fc="b", ec="b", zorder=10)
				c2 = Circle((target_x2[t], target_y2[t]), r, fc="r", ec="r", zorder=10)
				ax.add_patch(c0)
				ax.add_patch(c1)
				ax.add_patch(c2)
				
				ax.plot([0, pred_x1[t], pred_x2[t]],
				        [0, pred_y1[t], pred_y2[t]],
				        lw=2,
				        ls="--",
				        c="k",
				        label=f"Prediction T: {t / pred.shape[1] * 100:.0f} \% {steps} Steps", )
				# Circles representing the anchor point of rod 1, and bobs 1 and 2.
				c0 = Circle((0, 0), r / 2, fc="k", zorder=10)
				c1 = Circle((pred_x1[t], pred_y1[t]), r, fc="b", ec="b", zorder=10)
				c2 = Circle((pred_x2[t], pred_y2[t]), r, fc="r", ec="r", zorder=10)
				ax.add_patch(c0)
				ax.add_patch(c1)
				ax.add_patch(c2)
				
				# The trail will be divided into ns segments and plotted as a fading line.
				ns = 20
				s = max_trail // ns
				
				for j in range(ns):
					imin = t - (ns - j) * s
					if imin < 0:
						continue
					imax = imin + s + 1
					# The fading looks better if we square the fractional length along the trail.
					alpha = (j / ns) ** 2
					ax.plot(target_x2[imin:imax], target_y2[imin:imax], c="r", solid_capstyle="butt", lw=2, alpha=alpha, )
					ax.plot(pred_x2[imin:imax], pred_y2[imin:imax], c="r", solid_capstyle="butt", lw=2, alpha=alpha, )
				
				# Centre the image on the fixed anchor point, and ensure the axes are equal
				
				ax.set_xlim(-L1 - L2 - r, L1 + L2 + r)
				ax.set_ylim(-L1 - L2 - r, L1 + L2 + r)
				ax.set_aspect("equal", adjustable="box")
				plt.axis("off")
				plt.legend(bbox_to_anchor=(0.6, 1.2))
				plt.tight_layout()
			# plt.text(f"T: {t/pred.shape[1]*100:.0f} \%")
			
			if title != "":
				fig.suptitle(title)
			
			plt.tight_layout()
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			plt.close(fig)
			figs += [image]
		
		assert len(figs) > 10, f"{len(figs)=}"
		if not os.path.exists(phd_path / "DiffSim/experiments/media"):
			os.makedirs(phd_path / "DiffSim/experiments/media")  # create dir if not existent
		imageio.mimsave(phd_path / f"DiffSim/experiments/media/DoublePendulum{title}.gif", figs, fps=10, )
	
	def __repr__(self):
		return f"DoublePendulum_DataModule: Train {self.train_data.shape} | Val {self.val_data.shape}"


class ThreeBodyProblem_DataModule(LightningDataModule):
	def __init__(self, hparams):
		super().__init__()
		
		"""Add own hparams to already existing hparams and check keys"""
		self.hparams.update(hparams.__dict__)
		assert all([key_ in self.hparams for key_ in hparams.__dict__.keys()])
		
		self.data_shape = (self.hparams.nbody_spatial_dims * 2 * self.hparams.nbody_num_bodies,)
		self.in_features = math.prod(self.data_shape)
		self.out_features = self.in_features
		
		self.prepare_data()
	
	@staticmethod
	def datamodule_args(parent_parser):
		parser = parent_parser.add_argument_group("ThreeBody_DataModule")
		
		parser.add_argument("--data_dt", type=float, default=0.05)
		parser.add_argument("--data_timesteps", type=int, default=3000)
		parser.add_argument("--train_traj_repetition", type=int, default=6000)
		parser.add_argument("--nbodies", type=int, default=3)
		
		# parser.add_argument('--num_workers', type=int, default=4 * torch.cuda.device_count() if torch.cuda.is_available() else 0)
		
		return parent_parser
	
	def example_trajectory(self):
		"""Example initial conditionns"""
		# q0 = Tensor([[-0.5, 0, 0, 0.5, 0, 0, 0, 1, 0]]).float()[:,:self.num_bodies*self.spatial_dimensions]
		# p0 = Tensor([[0.01, 0.01, 0, -0.05, 0, -0.1, 0, -0.01, 0]]).float()[:,:self.num_bodies * self.spatial_dimensions]
		
		"""Assure that velocity/momentum values are not too small by not letting them fall below a value below 0.1 in any direction (pos or neg)"""
		q0 = torch.randn(size=(1, self.spatial_dimensions * self.num_bodies))
		q0 = torch.where(q0.abs() < 0.25, torch.sign(q0) * torch.ones_like(q0) * 0.25, q0).clamp(-0.5, 0.5)
		p0 = torch.randn_like(q0) * 0.1
		p0 = torch.where(p0.abs() < 0.1, torch.sign(p0) * torch.ones_like(p0) * 0.1, p0).clamp(-0.5, 0.5)
		
		x0 = torch.cat([q0, p0], dim=-1)
		
		diffeqs = VectorField(ThreeBodyProblemDiffEq(hparams=self.hparams), ThreeBodyProblem_ContractingForceField(), )
		# diffeqs = DiffEqIterator(self.threebodyproblem_diffeq)
		t = torch.linspace(start=0, end=300, steps=300) * self.hparams.data_dt
		
		diffeq = NeuralODE(diffeqs, sensitivity="adjoint", solver="rk4")
		
		with torch.inference_mode():
			traj = diffeq.trajectory(x0, t).swapdims(0, 1)
		
		assert traj.dim() == 3
		q, p = torch.chunk(traj, chunks=2, dim=-1)
		r = torch.chunk(q[0], chunks=self.num_bodies, dim=-1)  # whatever q is, chunk it into the number of bodies that we have
		r = [r_.numpy() for r_ in r]
		
		assert len(r) == self.hparams.nbody_num_bodies
		
		figs = []
		for t in tqdm(range(traj.shape[1])[1::5], desc="Creating GIF"):
			# Create figure
			fig = plt.figure(figsize=(15, 15))
			# Create 3D axes
			ax = fig.add_subplot(111, projection="3d" if self.hparams.nbody_spatial_dims == 3 else None)
			# Plot the orbits
			# ax.plot(r[:t,0], r[:t,1], r[:t, 2], color=color)
			for r_, color in zip(r, ["darkblue", "tab:red", "tab:green"]):
				ax.plot(*r_[:t].T, color=color)  # r.shape=[T,Dim] -> *[Dim, T]
				ax.scatter(*r_[t].T, color=color, marker="o", s=100)
			# ax.plot(r1_sol[:t, 0], r1_sol[:t, 1], r1_sol[:t, 2], color="darkblue")
			# ax.plot(r2_sol[:t, 0], r2_sol[:t, 1], r2_sol[:t, 2], color="tab:red")
			# ax.plot(r3_sol[:t, 0], r3_sol[:t, 1], r3_sol[:t, 2], color="tab:green")
			# Plot the final positions of the stars
			# ax.scatter(r1_sol[t, 0], r1_sol[t, 1], r1_sol[t, 2], color="darkblue", marker="o", s=100,label="Alpha Centauri A")
			# ax.scatter(r2_sol[t, 0], r2_sol[t, 1], r2_sol[t, 2], color="tab:red", marker="o", s=100,label="Alpha Centauri B")
			# ax.scatter(r3_sol[t, 0], r3_sol[t, 1], r3_sol[t, 2], color="tab:green", marker="o", s=100,label="Alpha Centauri C")
			# Add a few more bells and whistles
			ax.set_xlabel("x-coordinate", fontsize=14)
			ax.set_ylabel("y-coordinate", fontsize=14)
			ax.set_zlabel("z-coordinate", fontsize=14) if self.hparams.nbody_spatial_dimensions == 3 else None
			ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
			# ax.legend(loc="upper left", fontsize=14)
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			plt.close(fig)
			figs += [image]
		
		assert len(figs) > 10, f"{len(figs)=}"
		imageio.mimsave("./ThreeBodyProblemExample.gif", figs, fps=10)  # exit('@ThreeBodyProblem_DataModule.example_trajectory()')
	
	def prepare_data(self) -> None:
		
		num_trajs = 1
		
		q0 = Tensor([[-0.5, 0, 0, 0.5, 0, 0, 0, 1, 0]]).float()
		p0 = Tensor([[0.01, 0.01, 0, -0.05, 0, -0.1, 0, -0.01, 0]]).float()
		
		"""Assure that velocity/momentum values are not too small by not letting them fall below a value below 0.1 in any direction (pos or neg)"""
		q0 = torch.randn(size=(num_trajs, self.hparams.nbody_spatial_dims * self.hparams.nbody_num_bodies,))
		q0 = torch.where(q0.abs() < 0.25, torch.sign(q0) * torch.ones_like(q0) * 0.25, q0).clamp(-0.5, 0.5)
		p0 = torch.randn_like(q0) * 0.1
		p0 = torch.where(p0.abs() < 0.1, torch.sign(p0) * torch.ones_like(p0) * 0.1, p0).clamp(-0.5, 0.5)
		
		x0 = torch.cat([q0, p0], dim=-1)
		
		diffeqs = VectorField(ThreeBodyProblemDiffEq(hparams=self.hparams),  # ThreeBodyProblem_ContractingForceField(),
		                      # ThreeBodyProblem_SidewaysForceField()
		                      )
		t = (torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps, ) * self.hparams.data_dt)
		
		diffeq = NeuralODE(diffeqs, sensitivity="adjoint", solver=self.hparams.odeint)
		
		with torch.inference_mode():
			traj = diffeq.trajectory(x0, t).swapdims(0, 1)
		
		self.time = t.unsqueeze(0).repeat((num_trajs, 1))
		self.rawdata = traj
		
		assert (self.time.shape == self.rawdata.shape[:2]), f"{self.time.shape=} vs {self.rawdata.shape}"
		assert self.rawdata.shape == (num_trajs, self.hparams.data_timesteps, self.hparams.nbody_spatial_dims * self.hparams.nbody_num_bodies * 2,)
	
	def setup(self, stage: Optional[str] = None) -> None:
		
		assert (self.rawdata.dim() == 3), f"Should be [DataSet, Timestep, Channel, H, W], but is {self.rawdata.shape=}"
		assert self.rawdata.shape[1] == self.hparams.data_timesteps
		assert (self.time.shape == self.rawdata.shape[:2]), f"{self.time.shape=} vs {self.rawdata.shape}"
		assert (self.rawdata.dim() == 3 and self.time.dim() == 2), f"{self.rawdata.shape=} vs {self.time.shape=}"
		
		self.data = self.rawdata
		self.data_mean = self.data.mean(dim=[0, 1]).unsqueeze(0)
		self.data_std = self.data.std(dim=[0, 1]).unsqueeze(0)
		# self.data = (self.data - self.data_mean) / self.data_std
		
		self.y_mean = self.data_mean
		self.y_std = self.data_std
		dy = self.data[:, 1:] - self.data[:, :-1]
		self.dy_mean = dy.mean(dim=[0, 1]).unsqueeze(0)
		self.dy_std = dy.std(dim=[0, 1]).unsqueeze(0)
		
		assert (self.y_mean.shape == self.y_std.shape == self.dy_mean.shape == self.dy_std.shape == torch.Size((1,
		                                                                                                        self.hparams.nbody_num_bodies * self.hparams.nbody_spatial_dims * 2))), f"{self.y_mean.shape=} vs {self.y_std.shape=} vs {self.dy_mean.shape=} vs {self.dy_std.shape=}"
		
		train_size = int(self.data.shape[1] * self.hparams.val_split)
		val_size = self.data.shape[1] - train_size
		
		self.train_data, self.val_data = torch.split(self.data, [train_size, val_size], dim=1)
		self.train_time, self.val_time = torch.split(self.time, [train_size, val_size], dim=1)
		
		self.data_train = DiffEq_TimeSeries_DataSet(data=self.train_data,
		                                            time=self.train_time,
		                                            other_data=None,
		                                            input_length=self.hparams.input_length,
		                                            output_length=self.hparams.output_length_train,
		                                            output_length_sampling=self.hparams.iterative_output_length_training,
		                                            output_length_increase=self.hparams.iterative_output_length_training_increase,
		                                            traj_repetition=self.hparams.train_traj_repetition,
		                                            sample_axis="timesteps" if self.hparams.train_traj_repetition == 1 else "trajs", )
		
		self.data_val = DiffEq_TimeSeries_DataSet(data=self.val_data,
		                                          time=self.val_time,
		                                          other_data=None,
		                                          input_length=self.hparams.input_length,
		                                          output_length=self.hparams.output_length_val,
		                                          output_length_sampling=False,
		                                          traj_repetition=1,
		                                          sample_axis="timesteps", )
		
		print(self.data_train)
		print(self.data_val)
	
	def collate_fn(self, batch):
		
		y0 = []
		t = []
		T = batch[0][2]
		y = []
		
		for sample in batch:
			y0_, t_, T_, y_, latent_data_ = sample
			
			y0 += [y0_]
			t += [t_]
			y += [y_]
		
		y0 = torch.stack(y0)
		t = torch.stack(t)
		y = torch.stack(y)
		
		return {"y0": y0, "t": t, "T": T, "y": y}
	
	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		
		dataloader = DataLoader(self.data_train,
		                        batch_size=self.hparams.batch_size,
		                        shuffle=True,
		                        num_workers=self.hparams.num_workers,
		                        collate_fn=self.collate_fn, )
		
		return dataloader
	
	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val,
		                  batch_size=self.hparams.batch_size * 2,
		                  num_workers=self.hparams.num_workers,
		                  shuffle=False,
		                  collate_fn=self.collate_fn, )
	
	def show_prediction(self, target, pred=None, ):
		
		pred = pred[:, :300]
		target = target[:, :300]
		
		assert target.dim() == 3
		if pred is not None:
			assert pred.dim() == 3
		
		if pred is not None:
			q_pred, p_pred = torch.chunk(pred, chunks=2, dim=-1)
			q_pred = torch.chunk(q_pred[0],
			                     chunks=self.hparams.nbody_num_bodies,
			                     dim=-1)  # whatever q is, chunk it into the number of bodies that we have
			q_pred = [q_.numpy() for q_ in q_pred]
		
		q_target, p_target = torch.chunk(target, chunks=2, dim=-1)
		q_target = torch.chunk(q_target[0],
		                       chunks=self.hparams.nbody_num_bodies,
		                       dim=-1)  # whatever q is, chunk it into the number of bodies that we have
		q_target = [q_.numpy() for q_ in q_target]
		
		figs = []
		for t in tqdm(range(target.shape[1])[1::5], desc="Creating GIF"):
			# Create figure
			fig = plt.figure(figsize=(15, 15))
			# Create 3D axes
			ax = fig.add_subplot(111, projection="3d" if self.hparams.nbody_spatial_dims == 3 else None)
			# Plot the orbits
			# if pred is not None:
			# 	ax.plot(r1_sol_pred[:t, 0], r1_sol_pred[:t, 1], r1_sol_pred[:t, 2], ls='--', label='Pred', color="darkblue")
			# 	ax.plot(r2_sol_pred[:t, 0], r2_sol_pred[:t, 1], r2_sol_pred[:t, 2], ls='--', label='Pred', color="tab:red")
			# 	ax.plot(r3_sol_pred[:t, 0], r3_sol_pred[:t, 1], r3_sol_pred[:t, 2], ls='--', label='Pred', color="tab:green")
			# 	# Plot the final positions of the stars
			# 	ax.scatter(r1_sol_pred[t, 0], r1_sol_pred[t, 1], r1_sol_pred[t, 2], color="darkblue", marker="o", s=100,
			# 			   label="Alpha Centauri A")
			# 	ax.scatter(r2_sol_pred[t, 0], r2_sol_pred[t, 1], r2_sol_pred[t, 2], color="tab:red", marker="o", s=100,
			# 			   label="Alpha Centauri B")
			# 	ax.scatter(r3_sol_pred[t, 0], r3_sol_pred[t, 1], r3_sol_pred[t, 2], color="tab:green", marker="o", s=100,
			# 			   label="Alpha Centauri C")
			
			# ax.plot(r1_sol_target[:t, 0], r1_sol_target[:t, 1], r1_sol_target[:t, 2], label='Target', color="darkblue")
			# ax.plot(r2_sol_target[:t, 0], r2_sol_target[:t, 1], r2_sol_target[:t, 2], label='Target', color="tab:red")
			# ax.plot(r3_sol_target[:t, 0], r3_sol_target[:t, 1], r3_sol_target[:t, 2], label='Target', color="tab:green")
			#
			# ax.scatter(r1_sol_target[t, 0], r1_sol_target[t, 1], r1_sol_target[t, 2], color="darkblue", marker="o", s=100,
			# 		   label="Alpha Centauri A")
			# ax.scatter(r2_sol_target[t, 0], r2_sol_target[t, 1], r2_sol_target[t, 2], color="tab:red", marker="o", s=100,
			# 		   label="Alpha Centauri B")
			# ax.scatter(r3_sol_target[t, 0], r3_sol_target[t, 1], r3_sol_target[t, 2], color="tab:green", marker="o", s=100,
			# 		   label="Alpha Centauri C")
			if pred is not None:
				for r_, color in zip(q_pred, ["darkblue", "tab:red", "tab:green"]):
					ax.plot(*r_[:t].T, color=color, ls="--", label="Pred")  # r.shape=[T,Dim] -> *[Dim, T]
					ax.scatter(*r_[t].T, color=color, marker="o", s=100)
			for r_, color in zip(q_target, ["darkblue", "tab:red", "tab:green"]):
				ax.plot(*r_[:t].T, color=color, label="Target")  # r.shape=[T,Dim] -> *[Dim, T]
				ax.scatter(*r_[t].T, color=color, marker="o", s=100)
			
			# Add a few more bells and whistles
			ax.set_xlabel("x-coordinate", fontsize=14)
			ax.set_ylabel("y-coordinate", fontsize=14)
			ax.set_zlabel("z-coordinate", fontsize=14) if self.hparams.nbody_spatial_dims == 3 else None
			ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
			ax.legend(loc="upper left", fontsize=14)
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			plt.close(fig)
			figs += [image]
		
		assert len(figs) > 10, f"{len(figs)=}"
		imageio.mimsave("./ThreeBodyProblemPrediction.gif", figs, fps=10)
	
	def __repr__(self):
		return f"ThreeBodyProblem_DataModule: {self.rawdata.shape}"


class NdHamiltonian_DataModule(LightningDataModule):
	"""
	Nd GMM where the log prob is the potential
	PyTorch Sampling Shapes: [n x MC sample shape, batch shape, event_shape...]
	batch shape[1], event_shape[num_gaussians, nd]
	"""
	
	def __init__(self, hparams):
		super().__init__()
		
		"""Add own hparams to already existing hparams and check keys"""
		self.hparams.update(hparams.__dict__)
		assert all([key_ in self.hparams for key_ in hparams.__dict__.keys()])
		
		pytorch_lightning.utilities.seed.reset_seed()
		
		self.nd = self.hparams.nd
		self.num_gaussians = self.hparams.num_gaussians
		
		"""Create potential for Hamiltonian"""
		self.potential = GMM(hparams)
		self.known_vectorfield = NdHamiltonianDiffEq(hparams, potential=self.potential)
		self.ndhamiltoniandiffeq = self.known_vectorfield
		self.vectorfield = VectorField(self.known_vectorfield,
		                               NdHamiltonianDiffEq_TimeDependent_2DCircularDiffEq(hparams),
		                               NdHamiltonianDiffEq_OriginDiffEq(hparams))
		
		print(f"\nDiffSim Generating Data:")
		print(self.vectorfield)
		print()
		
		self.time_dependent = True if self.vectorfield.time_dependent else False
		
		self.data_shape = (2 * self.hparams.nd,)
		self.in_features = math.prod(self.data_shape)
		self.out_features = self.in_features
	
	def vectorfield_viz_input(self, only_q=True):
		
		viz_res = 21
		points = torch.meshgrid(self.hparams.nd * [torch.linspace(-2, 2, steps=viz_res)], indexing="ij")
		points = torch.stack([points_ for points_ in points], dim=-1)
		if not only_q:
			points = torch.cat([points, torch.zeros_like(points)], dim=-1)
		t = torch.linspace(0, self.hparams.data_timesteps, self.hparams.data_timesteps) * self.hparams.data_dt
		return points, t
	
	def compare_vectorfield(self, model_vectorfield):
		
		x, t = self.vectorfield_viz_input(only_q=False)  # x:[X, Y, 4], t:[T]
		if self.vectorfield.time_dependent:
			for name, module in self.vectorfield.named_modules():
				if isinstance(module, DiffEq) and module.time_dependent:
					module.t0 = torch.zeros(x.shape[0], 1)
			
		data_vf = self.vectorfield(x=x, t=t)
		
		if model_vectorfield.time_dependent:
			for name, module in model_vectorfield.named_modules():
				if isinstance(module, DiffEq) and module.time_dependent:
					module.t0 = torch.zeros(x.shape[0], 1)
		model_vf = model_vectorfield(x=x, t=t)
		
		print()
		print()
		exit('@ndhamiltonian compare vectorfield')
	
	def viz_vectorfields(self, vectorfield=None, trajs=None, t=None, path_suffix='Data'):
		
		'''Takes arbitrary vector fields which work with the provided datamodule example data and trajs and time and plots them'''
		
		if vectorfield == None:
			vectorfield = self.vectorfield
			
		# if hasattr(self, 'rawdata'):
		# 	trajs = self.rawdata
		# 	t = self.time
		
		if self.hparams.nd == 2:
			path_to_save_viz = "."
			"""q: only position, x=[q,p]"""
			if trajs is None and t is None:
				q, t = self.vectorfield_viz_input(only_q=True)  # x:[X, Y, 2], t:[T]
				x, t = self.vectorfield_viz_input(only_q=False)  # x:[X, Y, 4], t:[T]
			elif trajs is not None and t is not None:
				q, _ = self.vectorfield_viz_input(only_q=True)  # x:[X, Y, 2], t:[T]
				x, _ = self.vectorfield_viz_input(only_q=False)  # x:[X, Y, 4], t:[T]
			
			if not vectorfield.time_dependent:
				fig, axs = plt.subplots(2, 2)
				axs = axs.flatten()
				for i, diffeq in enumerate(vectorfield.diffeqs):
					diffeq.visualize(x, ax=axs[i])
				
				if trajs is not None:
					assert trajs.shape[1] == t.numel(), f"{trajs.shape=} {t.shape=}"
					plot_num_trajs = min(3, trajs.shape[0])
					colors = ['red', 'blue', 'green']
					for traj_i in range(plot_num_trajs):
						axs[0].scatter(trajs[traj_i, 0, 0], trajs[traj_i, 0, 1], c=colors[traj_i])
						axs[0].plot(trajs[traj_i, :, 0], trajs[traj_i, :, 1], c=colors[traj_i])
				
				plt.show()
			
			if vectorfield.time_dependent:
				num_figs = 40
				
				figs = []
				every_nth_plot = [t.shape[0] // num_figs, 2][0]
				for t_idx, t_ in tqdm(enumerate(t), desc='Plotting'):
					if t_idx % every_nth_plot != 0:
						continue  # skipping if it's not every_nth_plot
					fig, axs = plt.subplots(2, 2)
					fig.suptitle(f"T: {t_:.1f}/{t[-1]}")
					
					axs = axs.flatten()
					for i, diffeq in enumerate(vectorfield.diffeqs):
						diffeq.visualize(x=x, t=t_, ax=axs[i])
					
					vectorfield.visualize(x=x, t=t_, ax=axs[-1])
					
					if trajs is not None:
						assert trajs.shape[1] == t.numel(), f"{trajs.shape=} {t.shape=}"
						plot_num_trajs = min(3, trajs.shape[0])
						colors = ['red', 'blue', 'green']
						for traj_i in range(plot_num_trajs):
							axs[0].scatter(trajs[traj_i, 0, 0], trajs[traj_i, 0, 1], c=colors[traj_i])
							axs[0].plot(trajs[traj_i, :t_idx, 0], trajs[traj_i, :t_idx, 1], c=colors[traj_i])
					
					plt.tight_layout()
					fig.canvas.draw()
					plt.close()
					image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
					image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
					figs += [image]
				
				if not os.path.exists(phd_path / "DiffSim/experiments/media/NdHamiltonian"):
					os.makedirs(phd_path / "DiffSim/experiments/media/NdHamiltonian")
				imageio.mimsave(f"./media/NdHamiltonian/NdHamiltonianTimeDependentNoiseField_{path_suffix}.gif", figs, fps=10, )
				# wandb.log({"example": wandb.Video(np.moveaxis(np.stack(figs, axis=0), -1,1), fps=4, format="gif")})
				wandb.log({path_suffix: wandb.Video(f"./media/NdHamiltonian/NdHamiltonianTimeDependentNoiseField_{path_suffix}.gif")})
				
	
	@staticmethod
	def datamodule_args(parent_parser, timesteps=2000, nd=2, num_gaussians=4, num_trajs=7, train_traj_repetition=100):
		parser = parent_parser.add_argument_group("ThreeBody_DataModule")
		
		parser.add_argument("--data_dt", type=float, default=0.01)
		parser.add_argument("--data_timesteps", type=int, default=timesteps)
		parser.add_argument("--train_traj_repetition", type=int, default=train_traj_repetition)
		parser.add_argument("--num_trajs", type=int, default=num_trajs)
		parser.add_argument("--val_split", type=float, default=0.8)
		parser.add_argument("--nd", type=int, default=nd)
		parser.add_argument("--num_gaussians", type=int, default=num_gaussians)
		
		# parser.add_argument('--num_workers', type=int, default=4 * torch.cuda.device_count() if torch.cuda.is_available() else 0)
		return parent_parser
	
	def example_trajectory(self):
		
		"""Example initial conditionns"""
		q0 = self.potential.sample((self.hparams.num_trajs,)) * 0.1
		"""
		H = E_pot + E_vel = prob + p^2/2m with m=1
		E_pot can realistically not be larger than 1
		For H=1, we have p = (H - E_pot).mul_(2).pow(0.5)
		Due to small errors in the probability, which is a substitute for the potential, the test_hamiltonian_energy doesn't quite add up to the final H
		Increasing the variance of the clusters decreases the error
		"""
		p0 = torch.randn((self.hparams.num_trajs, self.hparams.nd)) * 0.0001
		p0 /= p0.pow(2).sum(dim=-1, keepdim=True).pow(0.5)  # norm to unit lengths
		E_pot = self.potential(q0)
		H = torch.ceil(E_pot)  # set total energy to next largest integer, difference is the kinetic energy
		
		vel = (H - E_pot).mul_(2.0).pow(0.5).unsqueeze(-1)
		p0 = p0 * vel
		x0 = torch.cat([q0, p0], dim=-1)
		hamiltonian_energy = self.potential(q0) + p0.pow(2).sum(-1) / 2
		
		t = torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps) * self.hparams.data_dt
		diffeq = TimeOffsetNeuralODE(vector_field=self.vectorfield, sensitivity="adjoint", solver="rk4")
		traj = diffeq.trajectory(x=x0.requires_grad_(), t_span=t).swapdims(0, 1).detach()
		
		assert traj.dim() == 3
		q, p = torch.chunk(traj, chunks=2, dim=-1)
		hamiltonian_energy = self.potential(q) + p.pow(2).sum(-1) / 2
		
		self.viz_vectorfields(trajs=traj, t=t, path_suffix='Data')
	
	def prepare_data(self) -> None:
		
		"""Example initial conditionns"""
		q0 = self.potential.sample((self.hparams.num_trajs,))
		"""
		H = E_pot + E_vel = prob + p^2/2m with m=1
		For H=1, we have p = (H - E_pot).mul_(2).pow(0.5)
		Unfortunately, the Hamiltonian is not constant
		"""
		p0 = torch.randn((self.hparams.num_trajs, self.hparams.nd))
		p0 /= p0.pow(2).sum(dim=-1, keepdim=True).pow(0.5)  # norm to unit lengths
		H, E_pot = 1, self.potential.log_prob(q0)
		
		vel = (H - E_pot).mul_(2.0).pow(0.5).unsqueeze(-1)
		p0 = p0 * vel
		x0 = torch.cat([q0, p0], dim=-1)
		hamiltonian_energy = self.potential.log_prob(q0) + p0.pow(2).sum(-1) / 2
		
		# diffeqs = VectorField(self.ndhamiltoniandiffeq)
		# diffeqs = self.vectorfield
		t = (torch.linspace(start=0, end=self.hparams.data_timesteps, steps=self.hparams.data_timesteps, ) * self.hparams.data_dt)
		diffeq = TimeOffsetNeuralODE(vector_field=self.vectorfield, sensitivity="adjoint", solver="rk4")
		traj = diffeq.trajectory(x=x0, t_span=t).swapdims(0, 1).detach()
		
		self.time = t.unsqueeze(0).repeat((self.hparams.num_trajs, 1))
		self.rawdata = traj
		
		assert (self.time.shape == self.rawdata.shape[:2]), f"{self.time.shape=} vs {self.rawdata.shape}"
		assert self.rawdata.shape == (self.hparams.num_trajs, self.hparams.data_timesteps,
		                              self.hparams.nd * 2,), f"{self.rawdata.shape=} vs {(self.hparams.num_trajs, self.hparams.data_timesteps, self.hparams.nd * 2)}"
	
	def setup(self, stage: Optional[str] = None) -> None:
		
		assert (self.rawdata.dim() == 3), f"Should be [DataSet, Timestep, Channel, H, W], but is {self.rawdata.shape=}"
		assert self.rawdata.shape[1] == self.hparams.data_timesteps
		assert (self.time.shape == self.rawdata.shape[:2]), f"{self.time.shape=} vs {self.rawdata.shape}"
		assert (self.rawdata.dim() == 3 and self.time.dim() == 2), f"{self.rawdata.shape=} vs {self.time.shape=}"
		
		self.data = self.rawdata
		self.data_mean = self.data.mean(dim=[0, 1]).unsqueeze(0)
		self.data_std = self.data.std(dim=[0, 1]).unsqueeze(0)
		# self.data = (self.data - self.data_mean) / self.data_std
		
		self.y_mean = self.data_mean
		self.y_std = self.data_std
		dy = self.data[:, 1:] - self.data[:, :-1]
		self.dy_mean = dy.mean(dim=[0, 1]).unsqueeze(0)
		self.dy_std = dy.std(dim=[0, 1]).unsqueeze(0)
		
		assert (self.y_mean.shape == self.y_std.shape == self.dy_mean.shape == self.dy_std.shape == torch.Size((1,
		                                                                                                        self.hparams.nd * 2))), f"{self.y_mean.shape=} vs {self.y_std.shape=} vs {self.dy_mean.shape=} vs {self.dy_std.shape=}"
		
		train_size = int(self.data.shape[1] * self.hparams.val_split)
		val_size = self.data.shape[1] - train_size
		
		self.train_data, self.val_data = torch.split(self.data, [train_size, val_size], dim=1)
		self.train_time, self.val_time = torch.split(self.time, [train_size, val_size], dim=1)
		
		self.data_train = DiffEq_TimeSeries_DataSet(data=self.train_data,
		                                            time=self.train_time,
		                                            other_data=None,
		                                            input_length=self.hparams.input_length,
		                                            output_length=self.hparams.output_length_train,
		                                            output_length_increase=self.hparams.iterative_output_length_training_increase,
		                                            traj_repetition=self.hparams.train_traj_repetition,
		                                            sample_axis="timesteps" if self.hparams.train_traj_repetition == 1 else "trajs", )
		
		self.data_val = DiffEq_TimeSeries_DataSet(data=self.val_data,
		                                          time=self.val_time,
		                                          other_data=None,
		                                          input_length=self.hparams.input_length,
		                                          output_length=self.hparams.output_length_val,
		                                          traj_repetition=1,
		                                          sample_axis="trajs", )
		
		print(self.data_train)
		print(self.data_val)
	
	def collate_fn(self, batch):
		assert type(batch[0]) == dict
		y0 = []
		t = []
		T = batch[0]['T']
		y = []
		
		for sample in batch:
			y0_, t_, T_, y_, latent_data_ = sample.values()
			
			y0 += [y0_]
			t += [t_]
			y += [y_]
		
		y0 = torch.stack(y0)
		t = torch.stack(t)
		y = torch.stack(y)
		
		return {"y0": y0, "t": t, "T": T, "y": y}
	
	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		
		dataloader = DataLoader(self.data_train,
		                        batch_size=self.hparams.batch_size,
		                        shuffle=True,
		                        num_workers=self.hparams.num_workers,
		                        collate_fn=self.collate_fn, )
		
		return dataloader
	
	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val,
		                  batch_size=self.hparams.batch_size * 2,
		                  num_workers=self.hparams.num_workers if torch.cuda.is_available() else 0,
		                  shuffle=False,
		                  collate_fn=self.collate_fn, )
	
	def __repr__(self):
		return f"ThreeBodyProblem_DataModule: {self.rawdata.shape}"


def load_dm_data(hparams):
	data_str = hparams.dataset
	if data_str in ["gaussianblob"]:
		dm = Circular_GaussianBlob_DataModule(hparams)
	elif data_str in ["forcefieldgaussianblob"]:
		dm = ForceField_Circular_GaussianBlob_DataModule(hparams)
	elif data_str in ["doublependulum"]:
		dm = DoublePendulum_DataModule(hparams)
	elif data_str in ["threebodyproblem"]:
		dm = ThreeBodyProblem_DataModule(hparams)
	elif data_str in ["ndhamiltonian"]:
		dm = NdHamiltonian_DataModule(hparams)
	else:
		exit(f"No valid dataset provided ...")
	
	# dm.prepare_data()
	# dm.setup()
	hparams.time_dependent = True if dm.time_dependent else False
	hparams.__dict__.update({"data_shape": dm.data_shape})
	hparams.__dict__.update({"latent_sim_dim": dm.data_shape})
	hparams.__dict__.update({"in_shape": dm.data_shape})
	hparams.__dict__.update({"out_shape": dm.data_shape})
	hparams.__dict__.update({"in_features": math.prod(dm.data_shape)})
	hparams.__dict__.update({"out_features": math.prod(dm.data_shape)})
	hparams.__dict__.update({"latent_sim_dim": dm.latent_sim_dim}) if hasattr(dm, "latent_sim_dim") else None
	hparams.__dict__.update({"latent_augment_dim": dm.latent_augment_dim}) if hasattr(dm, "latent_augment_dim") else None
	
	return dm


if __name__ == "__main__":
	from DiffSim import HParamParser
	
	hparams = HParamParser(dataset="forcefieldgaussianblob", data_timesteps=50, data_dt=0.01, data_radius=5)
	dm = load_dm_data(hparams)
	
	pass
