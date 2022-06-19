import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# import moviepy, imageio, imageio_ffmpeg, moviepy.editor
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning.trainer.supporters
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torchtyping import TensorType, patch_typeguard
# from typeguard import typechecked
# patch_typeguard()


file_path = Path(__file__).absolute()
cwd = file_path.parent
phd_path = file_path
for _ in range(len(cwd.parts)):
	phd_path = phd_path.parent
	if phd_path.parts[-1] == "PhD":
		break

sys.path.append(phd_path)

from DiffSim.src.DiffSim_CallBacks import CustomTQDMProgressBar
from DiffSim.src.DiffSim_DataModules import load_dm_data, DoublePendulum_DataModule, ThreeBodyProblem_DataModule, \
	NdHamiltonian_DataModule
from DiffSim.src.DiffSim_HyperparameterParser import process_hparams
from DiffSim.src.DiffSim_Models import LatentODE, DoublePendulumModel, ThreeBodyModel, NdHamiltonianModel
from DiffSim.src.DiffSim_HyperparameterParser import str2bool


fontsize = 30
params = {
 'font.size'       : fontsize,
 'legend.fontsize' : fontsize,
 'xtick.labelsize' : fontsize,
 'ytick.labelsize' : fontsize,
 'axes.labelsize'  : fontsize,
 'figure.figsize'  : (10, 10),
 'text.usetex'     : True,
 'mathtext.fontset': 'stix',
 'font.family'     : 'STIXGeneral'}

plt.rcParams.update(params)

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer


class PretrainModule(LightningModule):
	'''
	For models that convert from image space to vector space, we pretrain the models to reconstruct the image space
	'''
	
	def __init__(self, hparams, model, **kwargs):
		super().__init__()
		self.hparams.update(hparams)
		self.model = model
	
	@staticmethod
	def pretrainer_args(parent_parser):
		parser = parent_parser.add_argument_group("Pretrainer")
		
		parser.add_argument('--pretrain_max_epochs', type=int, default=50)
		parser.add_argument('--pretrain_lr', type=float, default=-1)
		return parent_parser
	
	def training_step(self, batch, batch_idx):
		'''

		:param batch: y0[BS, 1, F...], t[BS, t], t[BS, t] T:int, y[BS, t, F...] Optional[mu[BS, t, 1, 2], covar[BS, t, 1, 2, 2]
		:param batch_idx:
		:return:
		'''
		loss_dict = self.model.pretrain_training_step(batch)
		
		return loss_dict
	
	def validation_step(self, batch, batch_idx):
		'''

		:param batch: y0[BS, 1, F...], t[BS, t], T:int, y[BS, t, F...] Optional[mu[BS, t, 1, 2], covar[BS, t, 1, 2, 2]
		:param batch_idx:
		:return:
		'''
		loss_dict = self.model.pretrain_training_step(batch)
		
		return loss_dict
	
	def validation_epoch_end(self, outputs) -> None:
		keys = outputs[0].keys()
		
		# val_loss = torch.stack([x['Val/Loss'] for x in outputs]).mean()
		epoch = {}
		for key in keys:
			epoch["Val/Epoch" + key] = sum([x[key] for x in outputs]) / len(outputs)
		
		self.log_dict(epoch)
	
	@torch.inference_mode()
	def on_fit_end(self) -> None:
		if self.hparams.model == 'ae_latentode' and False:
			
			batch = next(iter(self.trainer.datamodule.val_dataloader()))
			pred = self.model(batch)
			self.trainer.datamodule.generate_gif(pred, batch['y'], t=batch['t'])
	
	def configure_optimizers(self):
		
		optim = torch.optim.Adam(self.parameters(),
		                         lr=1e-3 if self.hparams.pretrain_lr < 0 else self.hparams.lr,
		                         # betas=[0.9, 0.99],
		                         weight_decay=1e-7)
		
		scheduler = {
		 "scheduler"        : ReduceLROnPlateau(optim,
		                                        mode="min",
		                                        factor=0.1,
		                                        threshold=1e-3,
		                                        cooldown=1,
		                                        patience=2,
		                                        min_lr=1e-6,
		                                        verbose=True, ),
		 "monitor"          : "Val/Epochloss",
		 "interval"         : "epoch",
		 "reduce_on_plateau": True,
		 "frequency"        : 1,
		 "strict"           : True, }
		
		return [optim], [scheduler]
	
	def get_trainer_kwargs(self, hparams):
		
		hparams_ = hparams if type(hparams) == dict else hparams.__dict__
		pretrainer_args = {
		 'callbacks'        : [EarlyStopping(monitor='Val/Epochloss', patience=5, mode='min'),
		                       CustomTQDMProgressBar(refresh_rate=10, prefix='Pretrain')],
		 'log_every_n_steps': 10,
		 'max_epochs'       : self.hparams.pretrain_max_epochs}
		
		pretrainer_args.update({key: value for key, value in hparams_.items() if key in ['fast_dev_run']})
		
		return pretrainer_args


class TrainModule(LightningModule):
	
	def __init__(self, known_diffeq, **kwargs):
		super().__init__()
		
		self.save_hyperparameters(ignore=['known_vectorfield'])
		
		if self.hparams.model == 'ae_latentode':
			self.model = LatentODE(self.hparams)  # self.model(torch.randn(size=(7,1, *self.hparams.in_dims)), torch.linspace(start=0,end=4,steps=5).unsqueeze(0) * self.hparams.data_dt, (torch.randn(7,4,1,2), torch.randn(7, 4, 2, 2)))
		elif self.hparams.model == 'doublependulum':
			self.model = DoublePendulumModel(self.hparams)
		elif self.hparams.model == 'threebodyproblem':
			self.model = ThreeBodyModel(self.hparams)
		elif self.hparams.model == 'ndhamiltonian':
			self.model = NdHamiltonianModel(self.hparams, known_diffeq)
		else:
			exit(f'Wrong model: {self.hparams.model}')
		
		print(f"\nDiffSim Model:")
		print(self.model)
		print()
	
	
	# wandb.watch(self.model, log='all', log_freq=50)
	
	@staticmethod
	def args(parent_parser, project='testing', logging='online', fast_dev_run=0, seed=12345, dataset='ndhamiltonian'):
		
		parser = parent_parser.add_argument_group("LightningModule")
		parser.add_argument("--logging", type=str, default=logging)
		parser.add_argument("--project", type=str, default=project)
		parser.add_argument("--experiment", type=str, default=None)
		parser.add_argument("--plot", type=str, default=True)
		parser.add_argument("--show", type=str, default=True)
		parser.add_argument("--fast_dev_run", type=str2bool, default=fast_dev_run)
		parser.add_argument("--seed", type=str2bool, default=seed)
		
		parser.add_argument("--dataset", type=str, default=dataset)
		
		return parent_parser
	
	@staticmethod
	def trainer_args(parent_parser, lr=1e-4, max_epochs=20, output_length_train=10, output_length_val=20):
		parser = parent_parser.add_argument_group("Trainer")
		
		parser.add_argument('--criterion', type=str, choices=['MSE', 'MAE', 'AbsE'], default='MAE')
		parser.add_argument('--max_epochs', type=int, default=max_epochs)
		parser.add_argument('--lr', type=float, default=lr)
		parser.add_argument('--num_workers',
		                    type=int,
		                    default=4 * torch.cuda.device_count() if torch.cuda.is_available() else 0)
		parser.add_argument('--early_stopping_patience', default=5)
		parser.add_argument("--batch_size", type=int, default=128)
		parser.add_argument('--input_length', type=int, default=1)
		
		parser.add_argument('--output_length', type=int, default=2)
		parser.add_argument('--output_length_train', type=int, default=output_length_train)
		parser.add_argument('--output_length_val', type=int, default=output_length_val)
		parser.add_argument('--iterative_output_length_training_increase', type=int, default=1)
		return parent_parser
	
	def on_fit_start(self):
		
		# self.trainer.datamodule.example_trajectory()
		self.trainer.datamodule.viz_vectorfields(vectorfield=self.trainer.datamodule.diffeq, path_suffix='Data')
		self.trainer.datamodule.viz_vectorfields(vectorfield=self.model.diffeq, path_suffix='Model_Untrained')
		
		# assert torch.allclose(self.trainer.datamodule.potential.loc, self.model.vectorfield.diffeqs[0].potential.loc)
		
		if self.hparams.pretraining:
			pretrain_module = PretrainModule(self.hparams, model=model.model)
			pretrainer = Trainer(**pretrain_module.get_trainer_kwargs(hparams))
			pretrainer.fit(pretrain_module, datamodule=dm)
	
	@torch.inference_mode()
	def forward(self, t, x):
		pass
	
	def training_step(self, batch, batch_idx):
		'''

		:param batch: y0[BS, 1, F...], t[t]
		:param batch_idx:
		:return:
		'''
		if self.hparams.model == 'ae_latentode':
			return self.model.training_step(batch)
		
		elif self.hparams.model in ['doublependulum', 'threebodyproblem', 'ndhamiltonian']:
			pred = self.model(batch)
			
			# pred.sum().backward()
			# exit(f'@training_step')
			# d = {k.split('.')[-1]:v.data.cpu().clone() for k, v in self.named_parameters()}
			# if not hasattr(self, 'tracking_sim_hparams'):
			# 	self.tracking_sim_hparams = {k: [v] for k,v in d.items()}
			# else:
			# 	for key in d:
			# 		self.tracking_sim_hparams[key] += [d[key]]
			loss, extra_loss = self.model.criterion(pred, batch)
			
			# self.log_dict({'Train/T': batch['T'], **d}, prog_bar=True)
			self.log_dict({'Train/T': batch['T']}, prog_bar=True)
			# return {'loss': loss, self.hparams.criterion: loss.detach(), 'T': batch['T']}
			scalar_params = {name: scalar.detach().numpy().item() for name, scalar in self.named_parameters() if
			                 scalar.numel() == 1}
			self.log_dict(scalar_params, prog_bar=False, on_step=True)
			return {'loss': loss, 'T': batch['T']}
	
	def training_epoch_end(self, outputs):
		keys = outputs[0].keys()
		
		# val_loss = torch.stack([x['Val/Loss'] for x in outputs]).mean()
		epoch = {}
		for key in keys:
			epoch["Train/" + key] = sum([x[key] for x in outputs]) / len(outputs)
		
		self.log_dict(epoch, prog_bar=True)
		
		self.trainer.datamodule.viz_vectorfields(vectorfield=self.model.diffeq, path_suffix=f'ModelTrained_Epoch{self.current_epoch}')
	
	def on_train_epoch_end(self, unused: Optional = None) -> None:
		
		if type(self.trainer.train_dataloader.dataset) == pytorch_lightning.trainer.supporters.CombinedDataset:
			self.trainer.train_dataloader.dataset.datasets.increase_output_length()
		else:
			self.trainer.train_dataloader.dataset.increase_output_length()
	
	@torch.enable_grad()
	def validation_step(self, batch, batch_idx):
		
		if self.hparams.model == 'ae_latentode':
			return self.model.training_step(batch)
		
		elif self.hparams.model in ['doublependulum', 'threebodyproblem', 'ndhamiltonian']:
			pred = self.model(batch)
			# vf_diff = self.trainer.datamodule.compare_vectorfield(self.model.vectorfield)
			loss, extra_loss = self.model.criterion(pred, batch)
			return {self.hparams.criterion: loss, 'Val/T': batch['T']}
	
	def validation_epoch_end(self, outputs):
		keys = outputs[0].keys()
		
		# val_loss = torch.stack([x['Val/Loss'] for x in outputs]).mean()
		epoch = {}
		for key in keys:
			if key not in ['Val/T']:
				epoch["Val/Epoch" + key] = sum([x[key] for x in outputs]) / len(outputs)
		
		self.log_dict(epoch, prog_bar=True)
	
	def configure_optimizers(self):
		if self.hparams.lr == 0:
			warnings.warn(f"Learning rate is {self.hparams.lr}")
		
		optim = torch.optim.Adam(self.model.parameters(), lr=1e-3 if self.hparams.lr <= 0 else self.hparams.lr)
		# for name, param in self.model.named_parameters():
		# 	print(f"{name}: {param}")
		schedulers = {
		 'scheduler'        : ReduceLROnPlateau(optim,
		                                        mode='min',
		                                        factor=0.5,
		                                        threshold=1e-3,
		                                        patience=3,
		                                        min_lr=1e-5,
		                                        verbose=True),
		 'monitor'          : 'Val/Epoch' + self.hparams.criterion,
		 'interval'         : 'epoch',
		 'reduce_on_plateau': True,
		 'frequency'        : 1,
		 'strict'           : False}
		
		return [optim], [schedulers]
	
	def on_before_optimizer_step(self, optimizer, optimizer_idx) -> None:
		
		# for name, param in self.model.named_parameters():
		# 	print(f"{name}: {param.grad}")
		pass
	
	def on_fit_end(self):
		
		# self.trainer.datamodule.example_trajectory()
		self.trainer.datamodule.viz_vectorfields(vectorfield=self.model.diffeq, path_suffix='Model_Trained')
	
	# self.model.viz_vector_fields(title='Trained \n', training_data=self.trainer.datamodule.train_data)
	# self.trainer.datamodule.viz_prediction(self.model, title='Trained')
	
	def callbacks(self):
		callbacks = []
		
		early_stop_callback = EarlyStopping(monitor="Val/Epoch" + self.hparams.criterion,
		                                    mode="min",
		                                    patience=self.hparams.early_stopping_patience,
		                                    min_delta=0.0,
		                                    verbose=False, )
		callbacks += [early_stop_callback, CustomTQDMProgressBar(refresh_rate=50 if torch.cuda.is_available() else 10,
		                                                         print_every_epoch=True)]
		
		return callbacks


hparams = argparse.ArgumentParser()
hparams = TrainModule.args(hparams,
                           project='initial_testing',
                           dataset='ndhamiltonian',
                           seed=-1,
                           fast_dev_run=0,
                           logging=['online', 'disabled'][1])
hparams = TrainModule.trainer_args(hparams, lr=1e-3, max_epochs=10, output_length_train=10)
hparams = PretrainModule.pretrainer_args(hparams)

temp_args, _ = hparams.parse_known_args()
if temp_args.dataset == 'doublependulum':
	hparams = DoublePendulum_DataModule.datamodule_args(hparams,
	                                                    timesteps=2000,
	                                                    dt=0.1,
	                                                    num_trajs=200,
	                                                    train_traj_repetition=1000)
	hparams = DoublePendulumModel.model_args(hparams)
elif temp_args.dataset == 'threebodyproblem':
	hparams = ThreeBodyProblem_DataModule.datamodule_args(hparams)
elif temp_args.dataset == 'ndhamiltonian':
	hparams = NdHamiltonian_DataModule.datamodule_args(hparams,
	                                                   num_trajs=5000,
	                                                   data_dt=0.01,
	                                                   nd=2,
	                                                   num_gaussians=4,
	                                                   timesteps=500,
	                                                   train_traj_repetition=5)
	hparams = NdHamiltonianModel.model_args(hparams)

hparams = process_hparams(hparams, print_hparams=True)

dm = load_dm_data(hparams)
model = TrainModule(**vars(hparams), known_diffeq=dm.analytical_diffeq)

# Trainer(num_sanity_val_steps=)
trainer = Trainer.from_argparse_args(hparams,
                                     enable_checkpointing=False,
                                     enable_model_summary=None,
                                     num_sanity_val_steps=3,
                                     callbacks=model.callbacks(),
                                     # val_check_interval=1.,
                                     gpus=1 if torch.cuda.is_available() else None,
                                     reload_dataloaders_every_n_epochs=True,
                                     # min_epochs=hparams.output_length_val//hparams.iterative_output_length_training_increase,
                                     # log_every_n_steps=10,
                                     )

trainer.fit(model=model, datamodule=dm)
