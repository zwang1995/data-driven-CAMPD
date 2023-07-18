# Created on 19 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Basic functionalities to support this project

import os
import sys
import time
import csv
import random
import pickle
import numpy as np
import torch


def assign_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shuffle_array(params, array):
    assign_seed(params["seed"])
    random.shuffle(array)
    return array


class Logger(object):
    def __init__(self, log_file, stream=sys.stdout):
        self.terminal = stream
        self.log_file = log_file
        with open(log_file, "w") as f:
            f.write("")

    def write(self, massage):
        self.terminal.write(massage)
        with open(self.log_file, "a") as f:
            f.write(massage)

    def flush(self):
        pass


def timestamp():
    print(f"\n ----------------- DATETIME {time.ctime(time.time())} ----------------- \n", flush=True)


def create_directory(path):
    try:
        os.makedirs(path)
    except:
        pass
    return path


def write_list_to_txt(path, row, method="w"):
    list = [str(i) for i in row]
    string = "; ".join(list + ["\n"])
    if method == "w":
        with open(path, "w") as f:
            f.write(string)
    if method == "a":
        with open(path, "a") as f:
            f.write(string)


def write_string_to_txt(path, string, method="w"):
    if method == "w":
        with open(path, "w") as f:
            f.write(string + "\n")
    if method == "a":
        with open(path, "a") as f:
            f.write(string + "\n")


def read_txt(path):
    list = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line.strip()
        list.append(line.split("; ")[:-1])  # [:-1] is used to remove the new line character "\n"
    return list


def write_csv(path, raw, method="w"):
    id = open(path, method, newline="")
    writer = csv.writer(id)
    writer.writerow(raw)
    id.close()


def pickle_save(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def RandomStaticSampling(Combine_loop, sampling_num_total, random_state=42):
    random.seed(random_state)
    return random.choices(Combine_loop, k=sampling_num_total)


def Euclidean_distance(vec, weights=None):
    if weights is None:
        dist = np.linalg.norm(vec)
    else:
        dist = np.sqrt(np.sum([x ** 2 * weight for (x, weight) in zip(vec, weights)]))
    return dist


def SequenceWithEndPoint(start, stop, step):
    return np.arange(start, stop + step, step)


def ListValue2Str(alist):
    return list(map(str, alist))


""" # Activation functions """


def identity(z):
    z = z.astype(float)
    return z


def elu(z):
    return z if z > 0 else np.exp(z) - 1


def sigmoid(z):
    z = z.astype(float)
    return (1 / (1 + np.exp(-z)))


def softplus(z):
    z = z.astype(float)
    return np.log(1 + np.exp(z))


def tanh(z):
    z = z.astype(float)
    return (1 - 2 / (np.exp(2 * z) + 1))  # (exp(2*z)-1)/(exp(2*z)+1)


def get_AC(func):
    ACs = {"elu": np.vectorize(elu), "sigmoid": sigmoid, "softplus": softplus, "tanh": tanh, "identity": identity}
    return ACs[func]
