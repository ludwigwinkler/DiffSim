import argparse
import numbers
import os
import sys
from pathlib import Path
import wandb, time

import pytorch_lightning.utilities.seed

file_path = Path(__file__).absolute()
cwd = file_path.parent
phd_path = file_path
for _ in range(len(cwd.parts)):
	phd_path = phd_path.parent
	if phd_path.parts[-1] == "PhD":
		break

sys.path.append(phd_path)

from pytorch_lightning.loggers.wandb import WandbLogger


def str2bool(v):
	if isinstance(v, bool):
		return v
	elif type(v) == str:
		if v.lower() in ("yes", "true", "t", "y", "1"):
			return True
		elif v.lower() in ("no", "false", "f", "n", "0"):
			return False
	elif isinstance(v, numbers.Number):
		assert v in [0, 1]
		if v == 1:
			return True
		if v == 0:
			return False
	else:
		raise argparse.ArgumentTypeError(f"Invalid Value: {type(v)}")


dataset_nicestr_dict = {
 "gaussianblob"          : "Gaussian Blob",
 "forcefieldgaussianblob": "Force Field Gaussian Blob",
 "doublependulum"        : "Double Pendulum",
 "threebodyproblem"      : "Three Body Problem",
 "ndhamiltonian"         : "Nd Hamiltonian", }

model_nicestr_dict = {"lstm": "LSTM", "bi_lstm": "Bi-LSTM", "ode": "NeuralODE", "bi_ode": "Bi-NeuralODE", }


def process_hparams(hparams, print_hparams=False):
	hparams = hparams.parse_args()
	
	hparams.output_length_train = (hparams.output_length if hparams.output_length_train == -1 else hparams.output_length_train)
	hparams.output_length_val = (hparams.output_length if hparams.output_length_val == -1 else hparams.output_length_val)
	
	assert hparams.output_length_val > 1 and hparams.output_length_train > 1
	
	if hparams.experiment is None:
		experiment_str = f"{str(hparams.dataset)}_Ttrain{str(hparams.output_length_train)}_Tval{str(hparams.output_length_val)}"
	else:
		experiment_str = f"{hparams.experiment}_{str(hparams.dataset)}_Ttrain{str(hparams.output_length_train)}_Tval{str(hparams.output_length_val)}"
	
	hparams.__dict__.update({"experiment": experiment_str})
	hparams.__dict__.update({"ckptname": str(hparams.dataset) + "_T" + str(hparams.output_length_train)})
	hparams.__dict__.update({"dataset_nicestr": dataset_nicestr_dict[hparams.dataset]})

	hparams.project = 'diffsim_' + hparams.project
	
	assert hparams.output_length >= 1
	assert hparams.output_length_train >= 1
	assert hparams.output_length_val >= 1
	
	if hparams.seed >= 0:
		pytorch_lightning.utilities.seed.seed_everything(hparams.seed)
	
	'''
	Logging
	'''
	if hparams.logging == 'disabled':
		os.environ['WANDB_MODE'] = 'disabled'
	while True:
		try:
			# wandb.init(settings=wandb.Settings(start_method="fork"))
			os.system(f'wandb sync --clean-force')
			os.system("wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5")
			os.environ["WANDB_DISABLE_CODE"] = "false"
			logger = WandbLogger(entity="ludwigwinkler", project=hparams.project, name=hparams.experiment, mode=hparams.logging)
			break
		except:
			print(f"Waiting 2 seconds ...")
			time.sleep(2)
	
	hparams.__dict__.update({"logger": logger})
	wandb.init()
	wandb.run.log_code(str(phd_path / 'DiffSim'))
	
	if print_hparams:
		[print(f"\t {key}: {value}") for key, value in sorted(hparams.__dict__.items())]
	
	return hparams
