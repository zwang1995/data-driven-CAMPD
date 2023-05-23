# Created at 05 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Functionalities related to Artificial Neural Network

import shutil
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from param_utils import *


def get_x_0(params, prop_scaler=None):
    task = params["task"]
    if task != "solvent": column_index = params["column_index"]
    if task == "solvent":
        x_0 = [[1.642080264, 99.1326, 1026.70949, 161.702975, 1.8902657]]  # 3.664060664,
    elif task == "process":
        if column_index == "T1":
            x_0 = [[75, 4, 3.5, 3]]
        elif column_index == "T2":
            x_0 = [[12, 0.6, 3.5, 3]]
        elif column_index == "T1T2":
            x_0 = [[75, 4, 3.5, 3, 12, 0.6, 3.5, 3]]
    elif task == "solvent_process":
        if column_index == "T1":
            x_0 = [[1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 75, 4, 3.5, 3]]
        elif column_index == "T2":
            x_0 = [[1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 12, 0.6, 3.5, 3]]
        elif column_index == "T1T2":
            x_0 = [
                [1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 75, 4, 3.5, 3, 12, 0.6, 3.5, 3]]

    if prop_scaler is None:  # surrogate modeling
        return x_0
    else:  # surrogate optimization
        x_0 = get_Tensor(prop_scaler.transform(x_0))
        return x_0


class Molecule_reader():
    def __init__(self, no, name, smiles, property, performance):
        self.no = no
        self.name = name
        self.smiles = smiles
        self.property = np.array(property)
        self.performance = np.array(performance)


def get_act(type):
    acts = {"elu": nn.ELU(), "identity": nn.Identity(), "relu": nn.ReLU(), "selu": nn.SELU(),
            "sigmoid": nn.Sigmoid(), "softplus": nn.Softplus(), "tanh": nn.Tanh()}
    return acts[type]


class FNN_model(nn.Module):
    def __init__(self, params, n_layer, n_neuron, act):
        nn.Module.__init__(self)
        self.params = params
        self.n_layer = n_layer
        n_in, n_hid, n_out = n_neuron
        if params["FNN_task"] != "regression":
            n_out = 2
        if self.n_layer >= 1:
            self.l1 = nn.Linear(n_in, n_hid)
        if self.n_layer >= 2:
            self.l2 = nn.Linear(n_hid, n_hid)
        if self.n_layer == 0:
            self.l3 = nn.Linear(n_in, n_out)
        else:
            self.l3 = nn.Linear(n_hid, n_out)
        self.act_func = get_act(act) if self.params["model_nonlinear"] else get_act("identity")
        self.out_act_func = nn.Identity(dim=1) if params["FNN_task"] != "regression" \
            else nn.Sigmoid() if self.params["out_sig"] else nn.Identity()

    def forward(self, input):
        if self.n_layer == 0:
            output = self.out_act_func(self.l3(input))
        else:
            output = self.act_func(self.l1(input))
            if self.n_layer >= 2:
                output = self.act_func(self.l2(output))
            output = self.out_act_func(self.l3(output))
        return output


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        # self.val_loss_min = np.Inf

        self.update = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.update = True
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.update = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.update = True
            self.best_loss = val_loss
            self.counter = 0


def save_model_structure(params, string):
    with open(params["model_stru_file"], "w") as f:
        f.write(string)
    # shutil.copy("fnn_utils.py", params["out_path"])


def get_Tensor(x):
    x = Variable(torch.Tensor(np.array(x, dtype=np.float64)))
    return x


def get_TensorLong(x):
    x = Variable(torch.Tensor(np.array(x)).type(torch.LongTensor))
    return x


def RMSELoss(f, y):
    return torch.sqrt(torch.mean((f - y) ** 2))


def fit_scaler(matrix):
    scaler = StandardScaler()
    scaler.fit(matrix)
    return scaler
