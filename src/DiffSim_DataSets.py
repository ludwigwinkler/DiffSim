import numbers
import os, sys
from typing import Tuple, Union
from pathlib import Path

# from torchtyping import TensorType, patch_typeguard
# from typeguard import typechecked
# patch_typeguard()

import torch

import numpy as np
import matplotlib.pyplot as plt

fontsize = 20
params = {
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "axes.labelsize": fontsize,
    "figure.figsize": (20, 10),
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
}

plt.rcParams.update(params)

from torch.utils.data import Dataset
import torch.nn.functional as F

file_path = Path(__file__).absolute()
cwd = file_path.parent
phd_path = file_path
for _ in range(len(cwd.parts)):
    phd_path = phd_path.parent
    if phd_path.parts[-1] == "PhD":
        break

sys.path.append(phd_path)

file_path = os.path.dirname(os.path.abspath(__file__)) + "/MD_DataUtils.py"
cwd = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Tensor = torch.Tensor
Scalar = torch.scalar_tensor


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


def to_np(_tensor):
    if _tensor is not None:
        assert isinstance(_tensor, torch.Tensor)
        return _tensor.cpu().squeeze().detach().numpy()
    else:
        return None


def traverse_path(full_path: str, depth: Union[numbers.Number, None] = None):
    # assert type(depth)==int and depth<=0, print(f"{depth=}")

    folders = full_path.split("/")
    depths = ["/".join(folders[:depth]) for depth in range(1, len(folders))]

    if depth is None or depth == 0:
        assert os.path.isdir(full_path)
        return full_path
    elif depth < 0:
        assert os.path.isdir(depths[depth])
        return depths[depth]


"""
How to sample from variable length sequences [seq1, seq2, seq3 ...]

Cleanest Way:
	Keep three timeseries in trajectory:
	 	Data: [T, F]
	 	ModelID: [T] i.e. [0,0,0,0,...,1,1,1,1,0,0,0,0] with 0:simmd and 1:mlmd
	 	Length of Segments, i.e. [300, 20, 40, 20, ... ] with provides us with the information to split into relevant segments
	 	data.split(length) => Tuple(*seqs)
	Split 
	Construct Sampler from lengths of individual sequences [ seq1_length, seq2_length, seq3_length ... ], i.e. [ 300, 20, 40, 20 ... ]
	Sampler gives an index proportional to the individual sequence lengths
	Sample starting point randomly from individual sequence
	
	Problem: 
	
Hacky Way:
	Chop up every sequence to a predefined length
"""


class VariableTimeSeries_DataSet(Dataset):
    def __init__(self, seqs, input_length=1, output_length=2, traj_repetition=1):

        assert type(seqs) == list
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1

        self.input_length = input_length
        self.output_length = output_length

        self.seqs = seqs
        self.traj_repetition = traj_repetition

    def __getitem__(self, idx):

        """ '
        DiffEq Interfaces append the solutions to the starting value such that:
        [y0] -> Solver(t) -> [y0 y1 y2 ... yt]

        Python indexing seq[t0:(t0+T)] includes start and excludes end
        seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
        """

        total_length = self.output_length + self.input_length

        """
		Many short timeseries
		"""
        idx = idx % len(
            self.seqs
        )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
        traj = self.seqs[idx]  # selecting trajectory

        assert total_length <= traj.shape[0], f"{total_length=} !<= {traj.shape[0]=}"
        t0 = (
            np.random.choice(traj.shape[0] - total_length)
            if traj.shape[0] > total_length
            else 0
        )  # selecting starting time in trajectory

        y0 = traj[
            t0 : (t0 + self.input_length)
        ]  # selecting corresponding startin gpoint
        target = traj[t0 : (t0 + total_length)]  # snippet of trajectory

        # assert target.shape[0]==total_length, f"{target.shape=} VS {total_length}"

        return y0, self.output_length, target

    def __len__(self):
        return len(self.seqs) * self.traj_repetition

    def __repr__(self):
        return f"VariableTimeSeries_DataSet: {[data_.shape[0] for data_ in self.seqs]}"


class BiDirectional_VariableTimeSeries_DataSet(Dataset):
    def __init__(self, seqs, input_length=1, output_length=2, traj_repetition=1):
        assert type(seqs) == list
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1

        self.input_length = input_length
        self.output_length = output_length

        self.seqs = seqs
        self.traj_repetition = traj_repetition

    def __getitem__(self, idx):
        """ '
        DiffEq Interfaces append the solutions to the starting value such that:
        [y0] -> Solver(t) -> [y0 y1 y2 ... yt]

        Python indexing seq[t0:(t0+T)] includes start and excludes end
        seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
        """

        total_length = self.output_length + 2 * self.input_length

        """
		Many short timeseries
		"""
        idx = idx % len(
            self.seqs
        )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
        traj = self.seqs[idx]  # selecting trajectory

        # if traj.shape[0] <= total_length:
        # 	print('hi')

        assert traj.shape[0] >= total_length, f"{total_length=} !<= {traj.shape[0]=}"
        t0 = (
            np.random.choice(traj.shape[0] - total_length)
            if traj.shape[0] > total_length
            else 0
        )  # selecting starting time in trajectory
        t1 = t0 + self.input_length + self.output_length

        y0 = traj[
            t0 : (t0 + self.input_length)
        ]  # selecting corresponding startin gpoint
        y1 = traj[t1 : (t1 + self.input_length)]
        target = traj[t0 : (t0 + total_length)]  # snippet of trajectory
        y0 = torch.cat([y0, y1], dim=0)

        # assert target.shape[0]==total_length, f"{target.shape=} VS {total_length}"

        return y0, self.output_length, target

    def __len__(self):
        return len(self.seqs) * self.traj_repetition

    def __repr__(self):
        return f"VariableTimeSeries_DataSet: {[data_.shape[0] for data_ in self.seqs]}"


class TimeSeries_DataSet(Dataset):
    def __init__(
        self,
        data,
        other_data: Tuple = None,
        input_length=1,
        output_length=2,
        output_length_sampling=False,
        traj_repetition=1,
        sample_axis=None,
        transform=None,
    ):

        # assert data.dim() == 3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1

        if sample_axis is not None:
            assert sample_axis in ["trajs", "timesteps"], f"Invalid axis ampling"

        self.input_length = input_length
        self.output_length = output_length
        self.output_length_sampling = output_length_sampling
        if output_length_sampling:
            self.output_length_tmp = 2

        if other_data is not None:
            assert all(
                [other_data_.shape[1] == data.shape[1] for other_data_ in other_data]
            )

        self.data = data
        self.other_data = other_data
        self.traj_repetition = traj_repetition

        if sample_axis is None:
            if (
                self.data.shape[0] * self.traj_repetition >= self.data.shape[1]
            ):  # more trajs*timesteps than timesteps
                self.sample_axis = "trajs"
            # print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
            elif (
                self.data.shape[0] * self.traj_repetition < self.data.shape[1]
            ):  # more timesteps than trajs
                self.sample_axis = "timesteps"
            # print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
            else:
                raise ValueError("Sample axis not defined in data set")

        elif sample_axis is not None and sample_axis in ["trajs", "timesteps"]:
            self.sample_axis = sample_axis

    def __getitem__(self, idx):

        """ '
        DiffEq Interfaces append the solutions to the starting value such that:
        [y0] -> Solver(t) -> [y0 y1 y2 ... yt]

        Python indexing seq[t0:(t0+T)] includes start and excludes end
        seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
        """

        output_length = (
            self.output_length
            if self.output_length_sampling is False
            else self.output_length_tmp
        )
        total_length = output_length + self.input_length

        if self.sample_axis == "trajs":
            """
            Many short timeseries
            """
            idx = (
                idx % self.data.shape[0]
            )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
            traj = self.data[idx]  # selecting trajectory

            assert (
                traj.shape[0] - total_length
            ) >= 0, f"Trajectory time dimension {traj.shape[0]} is smaller than output length {output_length}"
            t0 = (
                np.random.choice(traj.shape[0] - total_length)
                if (traj.shape[0] - total_length) > 0
                else 0
            )  # selecting starting time in trajectory

            if self.other_data is not None:
                other_data = tuple(
                    [other_data_i[idx] for other_data_i in self.other_data]
                )

        elif self.sample_axis == "timesteps":
            """
            Few short timeseries
            """

            traj_index = np.random.choice(
                self.data.shape[0]
            )  # Randomly select one of the few timeseries
            traj = self.data[traj_index]  # select the timeseries

            t0 = idx % (
                self.data.shape[1] - total_length
            )  # we're sampling from the timesteps

            if self.other_data is not None:
                other_data = tuple(
                    [other_data_i[traj_index] for other_data_i in self.other_data]
                )

        y0 = traj[
            t0 : (t0 + self.input_length)
        ]  # selecting corresponding startin gpoint
        target = traj[t0 : (t0 + total_length)]  # snippet of trajectory

        if self.other_data is not None:
            other_data = tuple(
                [other_data_i[t0 : (t0 + total_length)] for other_data_i in other_data]
            )

        if self.other_data is None:
            return y0, output_length, target, None
        else:
            return y0, output_length, target, other_data

    def increase_output_length(self):
        self.output_length_tmp = (
            self.output_length_tmp + 2
            if self.output_length_tmp + 2 <= self.output_length
            else self.output_length
        )

    def __repr__(self):
        return f"TimSeries DataSet: {self.data.shape} Sampling: {self.sample_axis}"

    def __len__(self):

        if self.sample_axis == "trajs":
            return self.data.shape[0] * self.traj_repetition
        elif self.sample_axis == "timesteps":
            return self.data.shape[1] - (self.input_length + self.output_length)
        else:
            raise ValueError("Sample axis not defined in data set not defined")


class DiffEq_TimeSeries_DataSet(Dataset):
    def __init__(
        self,
        data,
        time,
        other_data: Tuple = None,
        input_length=1,
        output_length=2,
        output_length_increase=2,
        traj_repetition=1,
        sample_axis=None,
    ):

        # assert data.dim() == 3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1
        assert data.dim() >= 3

        if sample_axis is not None:
            assert sample_axis in ["trajs", "timesteps"], f"Invalid axis ampling"

        self.input_length = input_length
        self.output_length = output_length
        self.output_length_increase = output_length_increase
        self.traj_repetition = traj_repetition

        if other_data is not None:
            assert all(
                [other_data_.shape[1] == data.shape[1] for other_data_ in other_data]
            )

        self.data = data
        self.time = time
        assert (
            self.data.shape[1] == self.time.shape[1]
        ), f"{self.data.shape=} vs {self.time.shape=}"
        self.other_data = other_data

        if sample_axis is None:
            if (
                self.data.shape[0] * self.traj_repetition >= self.data.shape[1]
            ):  # more trajs*timesteps than timesteps
                self.sample_axis = "trajs"
            # print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
            elif (
                self.data.shape[0] * self.traj_repetition < self.data.shape[1]
            ):  # more timesteps than trajs
                self.sample_axis = "timesteps"
            # print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
            else:
                raise ValueError("Sample axis not defined in data set")

        elif sample_axis is not None and sample_axis in ["trajs", "timesteps"]:
            self.sample_axis = sample_axis

    def __getitem__(self, idx):

        """ '
        DiffEq Interfaces append the solutions to the starting value such that:
        [y0] -> Solver(t) -> [y0 y1 y2 ... yt]

        Python indexing seq[t0:(t0+T)] includes start and excludes end
        seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
        """

        output_length = self.output_length
        total_length = output_length + self.input_length

        if self.sample_axis == "trajs":
            """
            Many short timeseries
            """
            idx = (
                idx % self.data.shape[0]
            )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
            traj = self.data[idx]  # selecting trajectory

            assert (
                traj.shape[0] - total_length
            ) >= 0, f"Trajectory time dimension {traj.shape[0]} is smaller than output length {output_length}"
            t0 = (
                np.random.choice(traj.shape[0] - total_length)
                if (traj.shape[0] - total_length) > 0
                else 0
            )  # selecting starting time in trajectory
            time = self.time[idx, t0 : (t0 + total_length)]

            if self.other_data is not None:
                other_data = tuple(
                    [other_data_i[idx] for other_data_i in self.other_data]
                )

        elif self.sample_axis == "timesteps":
            """
            Few short timeseries
            """

            traj_index = np.random.choice(
                self.data.shape[0]
            )  # Randomly select one of the few timeseries
            traj = self.data[traj_index]  # select the timeseries

            t0 = idx % (
                self.data.shape[1] - total_length
            )  # we're sampling from the timesteps
            time = self.time[traj_index, t0 : (t0 + total_length)]

            if self.other_data is not None:
                other_data = tuple(
                    [other_data_i[traj_index] for other_data_i in self.other_data]
                )

        y0 = traj[
            t0 : (t0 + self.input_length)
        ]  # selecting corresponding startin gpoint
        target = traj[t0 : (t0 + total_length)]  # snippet of trajectory

        if self.other_data is not None:
            other_data = tuple(
                [other_data_i[t0 : (t0 + total_length)] for other_data_i in other_data]
            )

        if self.other_data is None:
            return {
                "y0": y0,
                "t": time,
                "T": output_length,
                "target": target,
                "other_data": None,
            }
        else:
            return {
                "y0": y0,
                "t": time,
                "T": output_length,
                "target": target,
                "other_data": other_data,
            }

    def increase_output_length(self):
        self.output_length += self.output_length_increase

    def __repr__(self):
        return f"DiffEq TimSeries DataSet: {self.data.shape} \t Sampling: {self.sample_axis} \t Traj Repetition:{self.traj_repetition}"

    def __len__(self):

        if self.sample_axis == "trajs":
            return self.data.shape[0] * self.traj_repetition
        elif self.sample_axis == "timesteps":
            return self.data.shape[1] - (self.input_length + self.output_length)
        else:
            raise ValueError("Sample axis not defined in data set not defined")


class BiDirectional_TimeSeries_DataSet(Dataset):
    def __init__(
        self,
        data,
        input_length=1,
        output_length=2,
        output_length_sampling=False,
        traj_repetition=1,
        sample_axis=None,
    ):

        assert (
            data.dim() == 3
        ), f"Data.dim()={data.dim()} and not [trajs, steps, features]"
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1

        if sample_axis is not None:
            assert sample_axis in ["trajs", "timesteps"], f"Invalid axis ampling"

        self.input_length = input_length
        self.output_length = output_length
        self.output_length_sampling = output_length_sampling
        self.output_length_samplerange = [1, output_length + 1]

        self.data = data
        self.traj_repetition = traj_repetition

        if sample_axis is None:
            if (
                self.data.shape[0] * self.traj_repetition >= self.data.shape[1]
            ):  # more trajs*timesteps than timesteps
                self.sample_axis = "trajs"
            # print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
            elif (
                self.data.shape[0] * self.traj_repetition < self.data.shape[1]
            ):  # more timesteps than trajs
                self.sample_axis = "timesteps"
            # print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
            else:
                raise ValueError("Sample axis not defined in data set")

        elif sample_axis is not None and sample_axis in ["trajs", "timesteps"]:
            self.sample_axis = sample_axis

    def sample_output_length(self):

        print(f"{self.sample_output_length=}")

        if self.output_length_sampling:
            self.sampled_output_length = np.random.randint(
                int(self.output_length_samplerange[0]),
                int(self.output_length_samplerange[1]),
            )

    def update_output_length_samplerange(self, low=0.1, high=0.5, mode="add"):

        assert mode in ["add", "set"], "mode is not set correctly"

        cur_low, cur_high = (
            self.output_length_samplerange[0],
            self.output_length_samplerange[1],
        )

        if mode == "add":
            if cur_high + high < self.__len__():
                cur_high += high
            if cur_low + low < cur_high:
                cur_low += low
            self.output_length_samplerange = np.array([cur_low, cur_high])
        elif mode == "set" and low < high:
            assert high < self.__len__()
            self.output_length_samplerange = np.array([low, high])
        else:
            raise ValueError("Incorrect inputs to update_batchlength_samplerange")

    def __getitem__(self, idx):

        if hasattr(self, "sampled_output_length"):
            output_length = self.sampled_output_length
        else:
            output_length = self.output_length

        total_length = output_length + 2 * self.input_length

        if self.sample_axis == "trajs":
            """
            Many short timeseries
            """
            idx = (
                idx % self.data.shape[0]
            )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
            traj = self.data[idx]  # selecting trajectory

            assert (
                traj.shape[0] - total_length
            ) >= 0, f" trajectory length {traj.shape[0]} is smaller than output_length {output_length}"
            t0 = np.random.choice(
                traj.shape[0] - total_length
            )  # selecting starting time in trajectory
            t1 = t0 + self.input_length + self.output_length

        elif self.sample_axis == "timesteps":
            """
            Few short timeseries
            """

            traj_index = np.random.choice(
                self.data.shape[0]
            )  # Randomly select one of the few timeseries
            traj = self.data[traj_index]  # select the timeseries

            t0 = idx % (
                self.data.shape[1] - total_length
            )  # we're sampling from the timesteps
            t1 = t0 + self.input_length + self.output_length
            assert (t0 + total_length) < self.data.shape[1]

        """
		y0 + input_length | output_length | y1 + input_length
		"""

        y0 = traj[
            t0 : (t0 + self.input_length)
        ]  # selecting corresponding starting point
        y1 = traj[
            t1 : (t1 + self.input_length)
        ]  # selecting corresponding starting point

        target = traj[t0 : (t0 + total_length)]  # snippet of trajectory

        assert y0.shape[0] == self.input_length
        assert y1.shape[0] == self.input_length
        assert target.shape[0] == total_length
        # assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
        # assert F.mse_loss(y1, target[-self.input_length:]) == 0, f'{F.mse_loss(y1[0], target[-1])=}'

        y0 = torch.cat([y0, y1], dim=0)

        return y0, output_length, target

    def __len__(self):

        # assert self.data.dim()==3

        if self.sample_axis == "trajs":
            return self.data.shape[0] * self.traj_repetition
        elif self.sample_axis == "timesteps":
            return self.data.shape[1] - (
                self.input_length + self.output_length + self.input_length
            )
        else:
            raise ValueError("Sample axis not defined in data set not defined")


class Sequential_BiDirectional_TimeSeries_DataSet(Dataset):
    def __init__(self, data, input_length=1, output_length=2, sample_axis=None):
        assert data.dim() == 2, f"Data.dim()={data.dim()} and not [steps, features]"
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1

        if sample_axis is not None:
            assert sample_axis in ["trajs", "timesteps"], f"Invalid axis ampling"

        self.input_length = input_length
        self.output_length = output_length

        T = data.shape[0]
        last_part = T % (input_length + output_length)
        data = data[: (T - last_part)]
        assert data.dim() == 2, f"{data.shape=}"

        data_ = torch.stack(
            data.chunk(chunks=T // (input_length + output_length), dim=0)[:-1]
        )  # dropping the last, possibly wrong length time series sample
        data_ = torch.cat([data_[:-1], data_[1:, :input_length]], dim=1)

        assert data_.shape[1] == (2 * input_length + output_length), f"{data_.shape=}"
        # plt.plot(data_[:3,:(-input_length),:3].flatten(0,1))
        # plt.show()
        assert (
            F.mse_loss(
                data_[:3, :(-input_length)].flatten(0, 1),
                data[: (3 * (input_length + output_length))],
            )
            == 0
        ), f"Data was not properly processed into segments"

        self.data = data_

    def __getitem__(self, idx):
        """
        Many short timeseries
        """
        idx = (
            idx % self.data.shape[0]
        )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
        traj = self.data[idx]  # selecting trajectory

        """
		y0 + input_length | output_length | y1 + input_length
		"""

        y0 = traj[: self.input_length]  # selecting corresponding starting point
        y1 = traj[-self.input_length :]  # selecting corresponding starting point

        target = traj  # snippet of trajectory
        assert (
            F.mse_loss(y0, target[: self.input_length]) == 0
        ), f"{F.mse_loss(y0, target[0])=}"
        assert (
            F.mse_loss(y1, target[-self.input_length :]) == 0
        ), f"{F.mse_loss(y1[0], target[-1])=}"

        y0 = torch.cat([y0, y1], dim=0)

        return y0, self.output_length, target

    def __len__(self):
        return self.data.shape[0]


class Sequential_TimeSeries_DataSet(Dataset):
    def __init__(self, data, input_length=1, output_length=2, sample_axis=None):

        assert data.dim() == 2, f"Data.dim()={data.dim()} and not [steps, features]"
        assert type(input_length) == int
        assert type(output_length) == int
        assert input_length >= 1
        assert output_length >= 1

        if sample_axis is not None:
            assert sample_axis in ["trajs", "timesteps"], f"Invalid axis ampling"

        self.input_length = input_length
        self.output_length = output_length

        # print(f"Sequential TimeSeriesData: {data.shape=}")

        T = data.shape[0]
        last_part = T % (input_length + output_length)
        data = data[: (T - last_part)]
        assert data.dim() == 2, f"{data.shape=}"
        data_ = torch.stack(
            data.chunk(chunks=T // (input_length + output_length), dim=0)
        )  # dropping the last, possibly wrong length time series sample

        assert data_.shape[1] == (input_length + output_length), f"{data_.shape=}"
        # plt.plot(data_[:3,:(-input_length),:3].flatten(0,1))
        # plt.show()
        assert (
            F.mse_loss(
                data_[:3].flatten(0, 1), data[: (3 * (input_length + output_length))]
            )
            == 0
        ), f"Data was not properly processed into segments"

        self.data = data_

    def __getitem__(self, idx):
        """
        Many short timeseries
        """
        idx = (
            idx % self.data.shape[0]
        )  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
        traj = self.data[idx]  # selecting trajectory

        """
		y0 + input_length | output_length
		"""

        y0 = traj[: self.input_length]  # selecting corresponding starting point

        target = traj  # snippet of trajectory
        assert (
            F.mse_loss(y0, target[: self.input_length]) == 0
        ), f"{F.mse_loss(y0, target[0])=}"
        # if self.input_length == 1: y0.squeeze_(0)

        return y0, self.output_length, target

    def __len__(self):
        return self.data.shape[0]
