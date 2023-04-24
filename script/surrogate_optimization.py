# Created at 19 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Data-driven optimization for Solvent Design, Process Optimization, and Integrated Solvent and Process Design

import pickle
import time

import joblib
import numpy as np
import pandas as pd
from basic_utils import *
from fnn_utils import *
from aspen_utils import *
# from viz_utils import *

from pymoo.config import Config

Config.show_compile_hint = False
from scipy.special import softmax
from pymoo.core.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import collections
from copy import deepcopy

np.seterr(divide="ignore", invalid="ignore")


def clean_screening_dataset(df_train, df_screen):
    train_sample_alias = df_train["alias"].tolist()

    df_screen = df_screen[df_screen["GAMMA_C4H8_Aspen"] != "None"]
    df_screen = df_screen[df_screen["GAMMA_C4H6_Aspen"] != "None"]
    df_screen = df_screen[df_screen["GAMMA_C4H8_Aspen"] != "ERROR"]
    df_screen = df_screen[df_screen["GAMMA_C4H6_Aspen"] != "ERROR"]
    df_screen = df_screen[~df_screen["GAMMA_C4H8_Aspen"].isnull()]
    df_screen = df_screen[~df_screen["GAMMA_C4H6_Aspen"].isnull()]
    df_screen = df_screen[df_screen["MW_Aspen"] != "ERROR"]
    df_screen = df_screen[df_screen["MUMX_Aspen"] != "None"]
    df_screen = df_screen[~df_screen["alias"].isin(train_sample_alias)]
    df_cand = df_screen
    df_cand = df_cand.reset_index(drop=True)

    df_screen = df_screen[df_screen["hasERROR"] == "FALSE"]
    df_screen = df_screen[df_screen["DIST_C4H8_T1"].astype(float) >= 0.995466491]
    df_screen = df_screen[df_screen["RebDuty_T1"].astype(float) >= 0]
    df_screen = df_screen[df_screen["RebDuty_T1"].astype(float) <= 16660.6957]
    df_ideal = df_screen
    df_ideal = df_ideal.reset_index(drop=True)

    return df_cand, df_ideal


def get_model_arch(task, model_spec, print_state=False):
    params = get_param(task, model_spec)
    perf = model_spec[0]

    n_in, n_out = len(params["prop_label"]), len(params["perf_label"])
    n_layer, n_hid, act = eval(read_txt(params["best_hyper_file"])[0][0])
    NLayer_dict[perf] = n_layer
    AC_dict[perf] = get_AC(act)
    OutSig_dict[perf] = params["out_sig"] if perf not in ["hasERROR_T1", "hasERROR_T2"] else None

    if print_state:
        print(f"-> Perf: {perf} / N_layer: {n_layer} / N_neuron: ({n_in}, {n_hid}, {n_out}) / Act: {act}", flush=True)

    model = FNN_model(params, n_layer, (n_in, n_hid, n_out), act)
    model.eval()
    del params
    return model


def InputExtractor(task, perf, x):
    if task == "process":
        x = x[:, :n_input_pro] if "T1" in perf else x[:, -n_input_pro:] if "T2" in perf else x
    if task == "solvent_process":
        x = x[:, :n_input_sol + n_input_pro] if "T1" in perf else np.hstack(
            (x[:, :n_input_sol], x[:, -n_input_pro:])) if "T2" in perf else x
    x = get_Tensor(x)
    return x


def get_model_params(model, task, model_spec, x_0, out_path, index, n_layer=1, return_params=True):
    perf, _, _ = model_spec

    params_dict = {}
    model_para_file = "".join([out_path, perf, "/model_", str(index), ".pt"])
    model.load_state_dict(torch.load(model_para_file))
    model.eval()

    if not return_params:
        return model
    else:
        x_0 = InputExtractor(task, perf, x_0)
        if perf == "RebDuty_T1":
            print("#", perf_scaler_T1.inverse_transform(model(x_0).detach().numpy())[0][0], flush=True)
        elif perf == "RebDuty_T2":
            print("#", perf_scaler_T2.inverse_transform(model(x_0).detach().numpy())[0][0], flush=True)
        # elif perf in ["hasERROR_T1", "hasERROR_T2"]:
        #     print("#", softmax(perf_scaler_T2.inverse_transform(model(x_0).detach().numpy()))[0][0], flush=True)
        else:
            print("#", model(x_0).detach().numpy()[0][0], flush=True)
        params_dict["w1"] = model.state_dict()["l1.weight"].detach().numpy()
        b1 = model.state_dict()["l1.bias"].detach().numpy()
        params_dict["b1"] = np.reshape(b1, (-1, 1))
        if n_layer == 2:
            params_dict["w2"] = model.state_dict()["l2.weight"].detach().numpy()
            b2 = model.state_dict()["l2.bias"].detach().numpy()
            params_dict["b2"] = np.reshape(b2, (-1, 1))
        params_dict["w3"] = model.state_dict()["l3.weight"].detach().numpy()
        b3 = model.state_dict()["l3.bias"].detach().numpy()
        params_dict["b3"] = np.reshape(b3, (-1, 1))
        return params_dict


def get_model_prediction(task, model_spec, X_for_pred):
    # X_for_pred: scaled, two-dimensional, without duplicating variables
    perf, FNN_task, column_index = model_spec
    pred_dict = {}

    for perf in Models:
        x_for_pred = InputExtractor(task, perf, X_for_pred)

        if "hasERROR" not in perf:
            model_spec = (perf, FNN_task, column_index)
        else:
            model_spec = (perf, "classification", column_index)
        params = get_param(task, model_spec)
        out_path = params["main_out_path"]
        f_list = []
        model = get_model_arch(task, model_spec)
        n_layer = NLayer_dict[perf]
        for i in range(5):
            model = get_model_params(model, task, model_spec, x_for_pred, out_path, i, n_layer, False)
            f = model(x_for_pred).detach().numpy()
            # if perf == "RebDuty":
            #     f = perf_scaler.inverse_transform(f)
            if perf == "RebDuty_T1":
                f = perf_scaler_T1.inverse_transform(f)
            elif perf == "RebDuty_T2":
                f = perf_scaler_T2.inverse_transform(f)
            elif perf in ["hasERROR_T1", "hasERROR_T2"]:
                f = softmax(f)
            f_list.append(f[0][0])
            pred_dict[perf] = np.array(f_list, dtype=np.float64)
        del params
    return pred_dict


class OptimProblem(Problem):
    def __init__(self, task, column_index, error_constr_state, solvent_bound=None, puri_constr=None):
        self.task = task
        self.column_index = column_index
        self.error_constr_state = error_constr_state
        self.solvent_bound = solvent_bound
        self.puri_constr = puri_constr

        def get_problem_setting(task, column_index):
            self.initial_state = True
            n_constr = 2
            if task == "solvent":
                xl, xu = [0.907091005, 70.09104, 712.218755, 125.947982, 0.337070321], \
                         [1.642181067, 200.32136, 1182.04039, 412.977047, 2.47090885]
                # [0.843961468, 71.1222, 701.186804, 114.732986, 0.226455082], \
                #          [1.642080264, 172.2676, 1618.32687, 395.4892, 8.43103044]
                n_obj = 2
            elif task == "process":
                if column_index == "T1":
                    xl, xu = [40, 1, 3.5, 1], [80, 10, 6, 8]
                    n_obj = 2
                    if error_constr_state: n_constr = 4  # g(x) <= 0
                elif column_index == "T2":
                    optimal_S_F = 2.38530065168425
                    xl, xu = [8, 0.2, 3.5, optimal_S_F], [20, 2, 6, optimal_S_F]
                    n_obj = 2
                    if error_constr_state: n_constr = 4  # g(x) <= 0
                elif column_index == "T1T2":
                    xl, xu = [40, 1, 3.5, 1, 8, 0.2, 3.5], [80, 10, 6, 8, 20, 2, 6]
                    n_obj = 2
                    n_constr = 3
                    if error_constr_state: n_constr = 5  # g(x) <= 0
            elif task == "solvent_process":
                if solvent_bound is not None:
                    solvent_l = solvent_bound * (1 - 1e-10)
                    solvent_u = solvent_bound * (1 + 1e-10)
                else:
                    solvent_l, solvent_u = [0.907091005, 1.32636452, 70.09104, 712.218755, 125.947982, 0.337070321], \
                                           [1.642181067, 5.944336533, 200.32136, 1182.04039, 412.977047, 2.47090885]
                if column_index == "T1":
                    xl, xu = solvent_l + [40, 1, 3.5, 1], solvent_u + [80, 10, 6, 8]
                    n_obj = 2
                    if error_constr_state: n_constr = 5  # g(x) <= 0
                elif column_index == "T2":
                    optimal_S_F = 2.38530065168425  # 3.22314120393114  # NMP: 2.38530065168425  # 2.404832714, 2.433799449
                    xl, xu = solvent_l + [8, 0.2, 3.5, optimal_S_F], solvent_u + [20, 2, 6, optimal_S_F]
                    n_obj = 2
                    if error_constr_state: n_constr = 5  # g(x) <= 0
                elif column_index == "T1T2":
                    xl, xu = np.concatenate((solvent_l, [40, 1, 3.5, 1, 8, 0.2, 3.5])), \
                             np.concatenate((solvent_u, [80, 10, 6, 8, 20, 2, 6]))
                    n_obj = 2
                    n_constr = 4
                    if error_constr_state: n_constr = 6  # g(x) <= 0
            return n_obj, n_constr, xl, xu

        n_obj, n_constr, xl, xu = get_problem_setting(task, column_index)
        print(f"-> N_obj: {n_obj} / N_constr: {n_constr} / X_low: {xl} / X_up: {xu}", flush=True)
        super().__init__(n_var=len(xl), n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        # X = np.array([[1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 75, 4, 3.5, 3, 12, 0.6, 3.5]]) # for test only
        f_list, g_list = [], []

        # if self.task == "solvent_process":
        #     f_list.append((X[:, -7] + X[:, -3]) / 100)

        """ # Input preprocessing """
        if self.task == "solvent":
            X = prop_scaler_T1.transform(X)
        elif self.task == "process":
            if self.column_index == "T1":
                X = prop_scaler_T1.transform(X)
            elif self.column_index == "T2":
                X = prop_scaler_T2.transform(X)
            elif self.column_index == "T1T2":
                X = np.concatenate((X, X[:, n_input_pro - 1].reshape(-1, 1)), axis=1)
                X = prop_scaler_T1T2.transform(X)
        elif self.task == "solvent_process":
            if self.column_index == "T1":
                X_std = prop_scaler_T1.transform(X)
            elif self.column_index == "T2":
                X_std = prop_scaler_T2.transform(X)
            elif self.column_index == "T1T2":
                X_std = np.concatenate((X, X[:, n_input_sol + n_input_pro - 1].reshape(-1, 1)),
                                       axis=1)  # dimension 6+4+4
                X_std = prop_scaler_T1T2.transform(X_std)
            X = X_std

        """ # Obtain scaler properties """
        if self.task == "solvent":
            perf_scaler_mean, perf_scaler_std = perf_scaler_T1.mean_[0], perf_scaler_T1.scale_[0]
        elif self.task == "process" or self.task == "solvent_process":
            f_duty_list, g_dutyL_list, g_dutyU_list = [], [], []
            if self.column_index == "T1":
                perf_scaler_mean_T1, perf_scaler_std_T1 = perf_scaler_T1.mean_[0], perf_scaler_T1.scale_[0]
            elif self.column_index == "T2":
                perf_scaler_mean_T2, perf_scaler_std_T2 = perf_scaler_T2.mean_[0], perf_scaler_T2.scale_[0]
            elif self.column_index == "T1T2":
                perf_scaler_mean_T1, perf_scaler_std_T1 = perf_scaler_T1.mean_[0], perf_scaler_T1.scale_[0]
                perf_scaler_mean_T2, perf_scaler_std_T2 = perf_scaler_T2.mean_[0], perf_scaler_T2.scale_[0]

        def get_f(x, params_dict, n_layer, ac, out_sig):
            w1, b1 = params_dict["w1"], params_dict["b1"]
            if n_layer == 2:
                w2, b2 = params_dict["w2"], params_dict["b2"]
            w3, b3 = params_dict["w3"], params_dict["b3"]
            if n_layer == 2:
                f = np.matmul(w3, ac(np.matmul(w2, ac(np.matmul(w1, x.T) + b1)) + b2)) + b3
            else:
                f = np.matmul(w3, ac(np.matmul(w1, x.T) + b1)) + b3
            if perf in OFs: f = np.sum(f, axis=0)
            f = sigmoid(f) if out_sig else f
            if perf in ["hasERROR_T1", "hasERROR_T2"]: f = softmax(f, axis=0)[0, :]
            return f

        for perf in Models:
            if self.task == "solvent":
                x = X
            elif self.task == "process":
                if "T1" in perf:
                    x = X[:, :n_input_pro]
                elif "T2" in perf:
                    x = X[:, -n_input_pro:]
            elif self.task == "solvent_process":
                if "T1" in perf:
                    x = X[:, :n_input_sol + n_input_pro]
                elif "T2" in perf:
                    x = np.concatenate((X[:, :n_input_sol], X[:, -n_input_pro:]), axis=1)
            f_ = []
            n_layer, ac, out_sig = NLayer_dict[perf], AC_dict[perf], OutSig_dict[perf]
            for params_dict in params_dict_list[perf]:
                f = get_f(x, params_dict, n_layer, ac, out_sig)
                f_.append(f)
            f_ = np.vstack(f_)

            # Task = "solvent"
            if self.task == "solvent":
                if perf == "DIST_C4H8_T1":
                    f_list.append(1 - np.average(f_, axis=0))
                    g_list.append(-np.average(f_, axis=0) + 0.965)  # purity >= 0
                    # g_list.append(np.average(f_, axis=0) - 1)  # purity <= 1
                elif perf in ["RebDuty_T1"]:
                    f_list.append(np.average(f_, axis=0))
                    # g_list.append((perf_scaler_mean + perf_scaler_std * np.average(f_, axis=0))-20000)  # duty >= 0

            # Task = "process"
            # elif self.task == "process":
            #     if perf == "DIST_C4H8_T1":
            #         f_list.append(1 - np.average(f_, axis=0))
            #         g_list.append(-np.average(f_, axis=0) + 0.98)  # purity >= 0
            #
            #     elif perf == "DIST_C4H6_T2":
            #         f_list.append(1 - np.average(f_, axis=0))
            #         g_list.append(-np.average(f_, axis=0) + 0.97)  # purity >= 0
            #         # g_list.append(np.average(f_, axis=0) - 1)  # purity <= 1
            #
            #     elif perf in ["RebDuty_T1", "RebDuty_T2"]:
            #         f_list.append(np.average(f_, axis=0))
            #         # g_list.append(-(perf_scaler_mean + perf_scaler_std * np.average(f_, axis=0)))  # duty >= 0
            #         g_list.append((perf_scaler_mean + perf_scaler_std * np.average(f_, axis=0)) - 3000)  # duty <= 0
            #
            #     elif perf in ["hasERROR_T1", "hasERROR_T2"]:
            #         if self.error_constr_state: g_list.append(np.average(f_, axis=0) - 0.5)

            # Task = "process"
            elif self.task == "process":
                if perf == "DIST_C4H8_T1":
                    f_list.append(1 - np.average(f_, axis=0))
                    g_list.append(-np.average(f_, axis=0) + 0.992)  # purity >= 0
                    if self.column_index == "T1":
                        f_list.append(1 - np.average(f_, axis=0))

                elif perf == "DIST_C4H6_T2":
                    g_list.append(-np.average(f_, axis=0) + 0.994)  # purity >= 0
                    if self.column_index == "T2":
                        f_list.append(1 - np.average(f_, axis=0))
                        f_list.append(np.average(f_, axis=0))

                elif perf in ["RebDuty_T1", "RebDuty_T2"]:
                    if perf == "RebDuty_T1":
                        mean_, std_ = perf_scaler_mean_T1, perf_scaler_std_T1
                        ori_f_ = mean_ + std_ * f_
                    elif perf == "RebDuty_T2":
                        mean_, std_ = perf_scaler_mean_T2, perf_scaler_std_T2
                        ori_f_ = mean_ + std_ * f_
                    # for ff in f_:
                    #     g_list.append(-(mean_ + std_ * ff))
                    f_duty_list.append(mean_ + std_ * np.average(f_, axis=0))
                    g_dutyU_list.append(np.average(ori_f_, axis=0))

                    # if self.column_index == "T1":
                    #     f_list.append(np.average(f_, axis=0))
                    #     g_list.append(np.average(f_, axis=0) - 20000)


                elif perf in ["hasERROR_T1", "hasERROR_T2"]:
                    if self.error_constr_state: g_list.append(np.average(f_, axis=0) - 0.5)


            # Task = "solvent_process"
            elif self.task == "solvent_process":
                if self.puri_constr is None:
                    puri_a, puri_b = 0.965, 0.985
                else:
                    puri_a, puri_b = self.puri_constr

                if perf == "DIST_C4H8_T1":
                    f_list.append(1 - np.average(f_, axis=0))
                    g_list.append(-np.average(f_, axis=0) + puri_a)  # constr >= 0
                    if self.column_index == "T1":
                        f_list.append(1 - np.average(f_, axis=0))

                elif perf == "DIST_C4H6_T2":
                    g_list.append(-np.average(f_, axis=0) + puri_b)  # constr >= 0
                    if self.column_index == "T2":
                        f_list.append(1 - np.average(f_, axis=0))
                        f_list.append(np.average(f_, axis=0))

                elif perf in ["RebDuty_T1", "RebDuty_T2"]:
                    if perf == "RebDuty_T1":
                        mean_, std_ = perf_scaler_mean_T1, perf_scaler_std_T1
                        ori_f_ = mean_ + std_ * f_
                        if self.solvent_bound is None: g_list.append(-2000 + np.std(ori_f_, axis=0))
                    elif perf == "RebDuty_T2":
                        mean_, std_ = perf_scaler_mean_T2, perf_scaler_std_T2
                        ori_f_ = mean_ + std_ * f_
                        if self.solvent_bound is None: g_list.append(-1000 + np.std(ori_f_, axis=0))
                    for ff in f_:
                        if self.solvent_bound is None: g_list.append(-(mean_ + std_ * ff))
                    f_duty_list.append(mean_ + std_ * np.average(f_, axis=0))
                    # g_dutyL_list.append(mean_ + std_ * np.average(f_, axis=0))
                    g_dutyU_list.append(np.average(ori_f_, axis=0))
                elif perf in ["hasERROR_T1", "hasERROR_T2"]:
                    if self.error_constr_state: g_list.append(np.average(f_, axis=0) - 0.5)

        if self.column_index == "T1T2":
            mean_joint = perf_scaler_T1T2.mean_
            std_joint = perf_scaler_T1T2.scale_

            f_list.append((np.sum(f_duty_list, axis=0) - mean_joint) / std_joint)
            # g_list.append(-np.sum(g_dutyL_list, axis=0))
            g_list.append(np.sum(g_dutyU_list, axis=0) - 35000)

        out["F"] = np.column_stack(f_list)
        out["G"] = np.column_stack(g_list)
        # print(out["F"], out["G"]) # for test only

        if self.initial_state:
            print(f"-> OF shape: {out['F'].shape}, Constraint shape: {out['G'].shape}")
            self.initial_state = False


# def get_x_0(params, prop_scaler):
#     task = params["task"]
#     if task != "solvent": column_index = params["column_index"]
#     if task == "solvent":
#         x_0 = [[1.642080264, 99.1326, 1026.70949, 161.702975, 1.8902657]]  # 3.664060664,
#     elif task == "process":
#         if column_index == "T1":
#             x_0 = [[75, 4, 3.5, 3]]
#         elif column_index == "T2":
#             x_0 = [[12, 0.6, 3.5, 3]]
#         elif column_index == "T1T2":
#             x_0 = [[75, 4, 3.5, 3, 12, 0.6, 3.5, 3]]
#     elif task == "solvent_process":
#         if column_index == "T1":
#             x_0 = [[1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 75, 4, 3.5, 3]]
#         elif column_index == "T2":
#             x_0 = [[1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 12, 0.6, 3.5, 3]]
#         elif column_index == "T1T2":
#             x_0 = [
#                 [1.642080264, 3.664060664, 99.1326, 1026.70949, 161.702975, 1.8902657, 75, 4, 3.5, 3, 12, 0.6, 3.5, 3]]
#     x_0 = get_Tensor(prop_scaler.transform(x_0))
#     return x_0


def get_OF_Model(task, column_index, class_state_T1=None, class_state_T2=None):
    if task == "solvent":
        Models = OFs = ["DIST_C4H8_T1", "RebDuty_T1"]
    elif task == "process" or task == "solvent_process":
        if column_index == "T1":
            OFs = ["DIST_C4H8_T1", "RebDuty_T1"]
        elif column_index == "T2":
            OFs = ["DIST_C4H6_T2", "RebDuty_T2"]
        elif column_index == "T1T2":
            OFs = ["DIST_C4H8_T1", "RebDuty_T1", "DIST_C4H6_T2", "RebDuty_T2"]

        if error_constr_state:
            if column_index == "T1":
                Models = OFs + ["hasERROR_T1"] if class_state_T1 else OFs
            if column_index == "T2":
                Models = OFs + ["hasERROR_T2"] if class_state_T2 else OFs
            if column_index == "T1T2":
                Models = OFs + [i for i, j in zip(["hasERROR_T1", "hasERROR_T2"], [class_state_T1, class_state_T2])
                                if j is True]
        else:
            Models = OFs
    return OFs, Models


def main(task, model_spec, nonlinear_state=True, error_constr=False):
    params_solvent = get_param("solvent", (None, "regression", "T1"))
    params_process = get_param("process", (None, "regression", "T1"))
    params_solvent_process = get_param("solvent_process", (None, "regression", "T1"))

    " # Define global variables "
    global n_input_sol, n_input_pro, error_constr_state, OFs, Models
    global prop_scaler_T1, perf_scaler_T1, prop_scaler_T2, perf_scaler_T2, prop_scaler_T1T2, perf_scaler_T1T2
    error_constr_state = error_constr

    _, FNN_task, column_index = model_spec

    " # Load settings "
    if task == "solvent":
        params = get_param(task, model_spec, nonlinear_state)
        n_input = len(params["prop_label"])
        prop_scaler_T1 = joblib.load(params["prop_scaler_file"])
        perf_scaler_T1 = joblib.load(params["perf_scaler_file"])
        x_0 = get_x_0(params, prop_scaler_T1)
        out_path = params["main_out_path"]

    elif task == "process":
        if column_index == "T1":
            params = params_T1 = get_param(task, ("RebDuty_T1", FNN_task, "T1"), nonlinear_state)
            n_input_pro = len(params["prop_label"])
            prop_scaler_T1 = joblib.load(params_T1["prop_scaler_file"])
            perf_scaler_T1 = joblib.load(params_T1["perf_scaler_file"])
            x_0 = get_x_0(params, prop_scaler_T1)
        elif column_index == "T2":
            params = params_T2 = get_param(task, ("RebDuty_T2", FNN_task, "T2"), nonlinear_state)
            n_input_pro = len(params["prop_label"])
            prop_scaler_T2 = joblib.load(params_T2["prop_scaler_file"])
            perf_scaler_T2 = joblib.load(params_T2["perf_scaler_file"])
            x_0 = get_x_0(params, prop_scaler_T2)
        elif column_index == "T1T2":
            params = get_param(task, ("RebDuty_T1", FNN_task, "T1T2"), nonlinear_state)
            n_input_pro = len(params["prop_label"])

            params_T1 = get_param(task, ("RebDuty_T1", FNN_task, "T1"), nonlinear_state)
            prop_scaler_T1 = joblib.load(params_T1["prop_scaler_file"])
            perf_scaler_T1 = joblib.load(params_T1["perf_scaler_file"])

            params_T2 = get_param(task, ("RebDuty_T2", FNN_task, "T2"), nonlinear_state)
            prop_scaler_T2 = joblib.load(params_T2["prop_scaler_file"])
            perf_scaler_T2 = joblib.load(params_T2["perf_scaler_file"])

            prop_scaler_T1T2 = StandardScaler()
            prop_scaler_T1T2.mean_ = np.concatenate((prop_scaler_T1.mean_, prop_scaler_T2.mean_))
            prop_scaler_T1T2.scale_ = np.concatenate((prop_scaler_T1.scale_, prop_scaler_T2.scale_))

            perf_scaler_T1T2 = StandardScaler()
            perf_scaler_T1T2.mean_ = perf_scaler_T1.mean_ + perf_scaler_T2.mean_
            perf_scaler_T1T2.scale_ = np.sqrt(perf_scaler_T1.var_ + perf_scaler_T2.var_)

            x_0 = get_x_0(params, prop_scaler_T1T2)
        out_path = params["main_out_path"]

    elif task == "solvent_process":
        n_input_sol = 6
        if column_index == "T1":
            params = params_T1 = get_param(task, ("RebDuty_T1", FNN_task, "T1"), nonlinear_state)
            n_input_pro = len(params["prop_label"]) - n_input_sol
            prop_scaler_T1 = joblib.load(params_T1["prop_scaler_file"])
            perf_scaler_T1 = joblib.load(params_T1["perf_scaler_file"])
            x_0 = get_x_0(params, prop_scaler_T1)
        elif column_index == "T2":
            params = params_T2 = get_param(task, ("RebDuty_T2", FNN_task, "T2"), nonlinear_state)
            n_input_pro = len(params["prop_label"]) - n_input_sol
            prop_scaler_T2 = joblib.load(params_T2["prop_scaler_file"])
            perf_scaler_T2 = joblib.load(params_T2["perf_scaler_file"])
            x_0 = get_x_0(params, prop_scaler_T2)
        elif column_index == "T1T2":
            params = get_param(task, ("RebDuty_T1", FNN_task, "T1T2"), nonlinear_state)
            n_input_pro = len(params["prop_label"]) - n_input_sol
            params_T1 = get_param(task, ("RebDuty_T1", FNN_task, "T1"), nonlinear_state)
            prop_scaler_T1 = joblib.load(params_T1["prop_scaler_file"])
            perf_scaler_T1 = joblib.load(params_T1["perf_scaler_file"])

            params_T2 = get_param(task, ("RebDuty_T2", FNN_task, "T2"), nonlinear_state)
            prop_scaler_T2 = joblib.load(params_T2["prop_scaler_file"])
            perf_scaler_T2 = joblib.load(params_T2["perf_scaler_file"])

            prop_scaler_T1T2 = StandardScaler()  # dimension: 6+4+4
            prop_scaler_T1T2.mean_ = np.concatenate((prop_scaler_T1.mean_, prop_scaler_T2.mean_[-n_input_pro:]))
            prop_scaler_T1T2.scale_ = np.concatenate((prop_scaler_T1.scale_, prop_scaler_T2.scale_[-n_input_pro:]))

            perf_scaler_T1T2 = StandardScaler()  # duty: T1+T2
            perf_scaler_T1T2.mean_ = perf_scaler_T1.mean_ + perf_scaler_T2.mean_
            perf_scaler_T1T2.scale_ = np.sqrt(perf_scaler_T1.var_ + perf_scaler_T2.var_)

            x_0 = get_x_0(params, prop_scaler_T1T2)
        out_path = params["main_out_path"]


    " # Specify output directory "
    char_ = "_constr" if error_constr_state else ""
    opt_path = create_directory(out_path + "optimization_" + column_index + char_ + "/")
    write_string_to_txt(opt_path + "history.txt", "")
    print(f"-> Nonlinear model & Constraints: {nonlinear_state}, {error_constr_state}", flush=True)

    if task == "solvent":
        OFs, Models = get_OF_Model(task, column_index)
    elif task == "process" or task == "solvent_process":
        params_ = get_param(task, ("hasERROR_T1", "classification", "T1"), nonlinear_state)
        class_state_T1, class_state_T2 = params_["class_train_state_T1"], params_["class_train_state_T2"]
        OFs, Models = get_OF_Model(task, column_index, class_state_T1, class_state_T2)
        del params_

    " # Load model parameters "
    global params_dict_list, NLayer_dict, AC_dict, OutSig_dict
    params_dict_list, NLayer_dict, AC_dict, OutSig_dict = {}, {}, {}, {}
    for perf in Models:
        params_dict_list[perf] = []
        model_spec_ = (perf, "regression", column_index) if perf not in ["hasERROR_T1", "hasERROR_T2"] \
            else (perf, "classification", column_index)
        params = get_param(task, model_spec_, nonlinear_state, False)
        model = get_model_arch(task, model_spec_, True)
        n_layer = NLayer_dict[perf]

        for i in range(5):
            params_dict = get_model_params(model, task, (perf, _, _), x_0, out_path, i, n_layer)
            params_dict_list[perf].append(deepcopy(params_dict))

    " # Define optimization problem "
    if task == "solvent":
        algorithm = NSGA2()
    elif task == "process" or task == "solvent_process":
        if task == "process":
            mask = ["int"] + ["real"] * (n_input_pro - 1) if column_index != "T1T2" \
                else ["int"] + ["real"] * (n_input_pro - 1) + ["int"] + ["real"] * (n_input_pro - 2)
        elif task == "solvent_process":
            mask = ["real"] * n_input_sol + ["int"] + ["real"] * (n_input_pro - 1) if column_index != "T1T2" \
                else ["real"] * n_input_sol + ["int"] + ["real"] * (n_input_pro - 1) + ["int"] + ["real"] * (
                    n_input_pro - 2)
        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "int": get_sampling("int_random")})
        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)})
        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm", eta=3.0),
            "int": get_mutation("int_pm", eta=3.0)})
        # termination = get_termination("n_eval", 200000)
        if task == "process":
            algorithm = NSGA2(sampling=sampling, crossover=crossover, mutation=mutation)
        elif task == "solvent_process":
            algorithm = NSGA2(pop_size=1000, sampling=sampling, crossover=crossover, mutation=mutation)
    problem = OptimProblem(task, column_index, error_constr_state)

    " # Perform multi-objective optimization "
    res = minimize(problem, algorithm, seed=params["seed"], save_history=True, verbose=True)
    np.save(opt_path + "X_best.npy", res.X)
    np.save(opt_path + "F_best.npy", res.F)
    pickle_save(opt_path + "optimization.pickle", res.history)
    print("-> Best solution found: %s" % res.X, flush=True)
    # print(res.X[:, :n_input_sol])
    # print(res.X[:, n_input_sol:])
    print("-> Function value: %s" % res.F, flush=True)
    # print("Constraint violation: %s" % res.CV)
    time.sleep(5)

    " # Solvent matching and process simulation "
    if task == "solvent":
        " # Prepare the solvent set for matching "
        df_train = pd.read_csv("../data/solvent/data_for_modeling.csv")
        df_screen = pd.read_csv("../data/solvent/data_all.csv")
        df_cand, df_ideal = clean_screening_dataset(df_train, df_screen)
        df_cand.to_csv(opt_path + "data_for_mapping.csv")
        df_ideal.to_csv(opt_path + "data_ideal.csv")

        print(f"-> Candidate & Ideal number: {len(df_cand)}, {len(df_ideal)}", flush=True)
        extra_alias = df_cand["alias"].tolist()
        extra_props = np.array(df_cand[params["prop_label"]].values)
        extra_perfs = np.array(df_cand[OFs].values)
        extra_simu_error = df_cand["hasERROR"].tolist()
        extra_std_props = prop_scaler_T1.transform(extra_props)

        ideal_alias = df_ideal["alias"].tolist()

        lb = [0.907091005, 70.09104, 712.218755, 125.947982, 0.337070321]  # 1.32636452,
        ub = [1.642181067, 200.32136, 1182.04039, 412.977047, 2.47090885]  # 5.944336533,

        for alias, std_prop, prop in zip(extra_alias, extra_std_props, extra_props):
            prop = [float(i) for i in prop]
            print(prop, std_prop)
            dist = Euclidean_distance(std_prop)
            # print(prop)
            # print([lbi < propi for lbi, propi in zip(lb, prop)], [propi < ubi for propi, ubi in zip(prop, ub)])
            state = np.array([np.array([lbi < propi for lbi, propi in zip(lb, prop)]).all(),
                  np.array([propi < ubi for propi, ubi in zip(prop, ub)]).all()]).all()
            f1, f2 = get_model_prediction(task, model_spec, get_Tensor(std_prop.reshape(1, -1))).values()
            print(alias, state, dist, np.mean(f1), np.std(f1), np.mean(f2), np.std(f2))

        time.sleep(10)

        " # Perform molecular mapping "
        optimal_indexes = []
        for x in res.X:
            x = prop_scaler_T1.transform(x.reshape(1, -1))
            print(x)
            distance_dict = {}
            for (index, prop) in enumerate(extra_std_props):
                dist = Euclidean_distance(x - prop)
                distance_dict[index] = dist
            sort_dist_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda x: x[1])[:5]}
            print(sort_dist_dict, flush=True)
            optimal_index = list(sort_dist_dict.keys())
            for i in optimal_index:
                optimal_indexes.append(i)
            # print([(extra_alias[i] in ideal_alias, extra_perfs[i]) for i in optimal_index])
        counter = collections.Counter(optimal_indexes)
        counter = collections.OrderedDict(counter.most_common())
        print("# Candidate occurrence:", dict(counter), flush=True)
        optimal_indexes = counter.keys()
        print(f"-> # candidates: {len(optimal_indexes)}", flush=True)
        print("-> Satisfied solvents:", np.sum([extra_alias[i] in ideal_alias for i in optimal_indexes]),
              [(i, extra_alias[i]) for i in optimal_indexes if extra_alias[i] in ideal_alias], flush=True)

        identified_cand_file = opt_path + "identified_candidate.csv"
        write_csv(identified_cand_file, ["index", "alias", "x", "x_std",
                                         "f_ave_C4H8", "f_ave_Duty", "f_std_C4H8", "f_std_Duty",
                                         "fs_C4H8", "fs_Duty", "DIST_C4H8", "RebDuty", "hasERROR", "is_Ideal"])
        for optimal_index in optimal_indexes:
            alias = extra_alias[optimal_index]
            success_state = alias in ideal_alias
            x, x_std = extra_props[optimal_index], extra_std_props[optimal_index]
            y = extra_perfs[optimal_index]
            simu_error = extra_simu_error[optimal_index]
            f1, f2 = get_model_prediction(task, model_spec, get_Tensor(x_std.reshape(1, -1))).values()
            write_csv(identified_cand_file, [optimal_index, alias, x, x_std,
                                             np.average(f1), np.average(f2), np.std(f1), np.std(f2),
                                             f1, f2, *y, simu_error, success_state], "a")

    elif task == "process":
        Combine_loop = res.X
        if column_index == "T1":
            Y = np.array([1 - res.F[:, 0], perf_scaler_T1.inverse_transform(res.F[:, 1])]).T
            Combine_loop_ = prop_scaler_T1.transform(Combine_loop)
        elif column_index == "T2":
            Y = np.array([1 - res.F[:, 0], perf_scaler_T2.inverse_transform(res.F[:, 1])]).T
            Combine_loop_ = prop_scaler_T2.transform(Combine_loop)
        elif column_index == "T1T2":
            # Y = np.array([1 - res.F[:, 0], 1 - res.F[:, 1], perf_scaler_T1T2.inverse_transform(res.F[:, 2])]).T
            Y = np.array([1 - res.F[:, 0], perf_scaler_T1T2.inverse_transform(res.F[:, 1])]).T
            Combine_loop_8D = np.hstack((Combine_loop, Combine_loop[:, n_input_pro - 1].reshape(-1, 1)))
            Combine_loop_ = prop_scaler_T1T2.transform(Combine_loop_8D)

        F = []
        for x_std in Combine_loop_:
            x_std = x_std.reshape(1, -1)
            if column_index != "T1T2":
                f_all = get_model_prediction(task, model_spec, get_Tensor(x_std)).values()
                F.append(np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F"))
            elif column_index == "T1T2":
                f_all = get_model_prediction(task, model_spec, get_Tensor(x_std)).values()
                F.append(np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F"))

        time.sleep(5)
        if (column_index == "T1") or (column_index == "T2"):
            from Aspen_c_ED import run_Simulation
        elif column_index == "T1T2":
            from Aspen_c_ED_T1T2 import run_Simulation
        TimeStart = time.time()

        total_run = len(Combine_loop)
        print(f"-> {total_run} simulations to be run", flush=True)
        run_Index = 0
        Outputs = []
        while run_Index < total_run - 1:
            # print("Connecting ...", flush=True)
            # try:
            txt_name = "DistillColumn.txt" if column_index == "T1" else "RecoveryColumn.txt" \
                if column_index == "T2" else "DistillRecoveryColumn.txt"
            if (column_index == "T1") or (column_index == "T2"):
                run_Index, output = run_Simulation(Combine_loop, column_index, opt_path + txt_name, run_Index)
            elif column_index == "T1T2":
                run_Index, output = run_Simulation(Combine_loop, opt_path + txt_name, run_Index)
            Outputs += output
            print(f"Deconnecting at #{run_Index}", flush=True)
            # except:
            #     print("Error detected ...", flush=True)

            KillAspen()
            time.sleep(10)

        TimeEnd = time.time()
        print(f"Lunch & Simulation time:{TimeEnd - TimeStart: .2f} s for {len(Combine_loop)} runs", flush=True)
        pred_columns = ["f_all_" + i for i in Models] + \
                       ["f_ave_" + i for i in Models] + \
                       ["f_std_" + i for i in Models]
        if column_index == "T1":
            columns = ["NStage_T1", "RR_T1", "TopPres_T1", "StoF"] + \
                      ["opt_DIST_C4H8_T1", "opt_RebDuty_T1"] + \
                      pred_columns + \
                      ["DIST_C4H8_T1", "RebDuty_T1", "hasIssue_T1", "hasERROR_T1"]
        elif column_index == "T2":
            columns = ["NStage_T2", "RR_T2", "TopPres_T2", "StoF"] + \
                      ["opt_DIST_C4H6_T2", "opt_RebDuty_T2"] + \
                      pred_columns + \
                      ["DIST_C4H6_T2", "RebDuty_T2", "hasIssue_T2", "hasERROR_T2"]
        elif column_index == "T1T2":
            columns = ["NStage_T1", "RR_T1", "TopPres_T1", "StoF", "NStage_T2", "RR_T2", "TopPres_T2"] + \
                      ["opt_DIST_C4H8_T1", "opt_RebDuty_total"] + \
                      pred_columns + \
                      ["DIST_C4H8_T1", "DIST_C4H6_T1", "RebDuty_T1",
                       "DIST_C4H6_T2", "DIST_SOL_T2", "RebDuty_T2",
                       "hasIssue", "hasERROR"]
        df = pd.DataFrame([ListValue2Str(c) + ListValue2Str(y) + ListValue2Str(f) + ListValue2Str(o)
                           for c, y, f, o in zip(Combine_loop, Y, F, Outputs)], columns=columns)
        if column_index == "T1T2":
            df["RebDuty_T1"] = df["RebDuty_T1"].astype(float)
            df["RebDuty_T2"] = df["RebDuty_T2"].astype(float)
            df["RebDuty_total"] = df["RebDuty_T1"] + df["RebDuty_T2"]
        df.to_csv(opt_path + "process_evaluation" + char_ + ".csv")

    elif task == "solvent_process":
        weights = [0.60171625, 0.44746581, 0.22374001, 0.54585274, 0.33455037]

        " # Prepare the solvent set for matching "
        df_train = pd.read_csv("../data/solvent/data_for_modeling.csv")
        df_screen = pd.read_csv("../data/solvent/data_all.csv")
        df_cand, df_ideal = clean_screening_dataset(df_train, df_screen)
        df_cand.to_csv(opt_path + "data_for_mapping.csv")
        df_ideal.to_csv(opt_path + "data_ideal.csv")

        prop_scaler_ = StandardScaler()
        if column_index == "T1" or column_index == "T1T2":
            prop_scaler_.mean_ = prop_scaler_T1.mean_[:n_input_sol]
            prop_scaler_.scale_ = prop_scaler_T1.scale_[:n_input_sol]
        elif column_index == "T2":
            prop_scaler_.mean_ = prop_scaler_T2.mean_[:n_input_sol]
            prop_scaler_.scale_ = prop_scaler_T2.scale_[:n_input_sol]

        print(f"-> Candidate & Ideal number: {len(df_cand)}, {len(df_ideal)}", flush=True)
        extra_alias = df_cand["alias"].tolist()
        extra_props = np.array(df_cand[params["prop_label"][:n_input_sol]].values).astype(np.float)
        extra_perfs = np.array(df_cand[OFs].values)
        extra_simu_error = df_cand["hasERROR"].tolist()
        extra_std_props = prop_scaler_.transform(extra_props)

        ideal_alias = df_ideal["alias"].tolist()

        " # Perform molecular mapping "
        for w in [None]:  # , weights
            optimal_indexes = []
            A, F = [], []
            for (sol_index, x) in enumerate(res.X):
                x_sol, x_pro = x[:n_input_sol], x[n_input_sol:]
                x_sol_std = prop_scaler_.transform(x_sol.reshape(1, -1))
                # print(x_sol, x_pro)
                distance_dict = {}
                for (index, prop) in enumerate(extra_std_props):
                    dist = Euclidean_distance(x_sol_std - prop, w)
                    distance_dict[index] = dist
                sort_dist_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda x: x[1])[:5]}
                # print(sort_dist_dict, flush=True)
                optimal_index = list(sort_dist_dict.keys())
                candidate_alias = [extra_alias[i] for i in optimal_index]
                candidate_props = [extra_props[i] for i in optimal_index]

                " # Process simulation under optimal operating conditions and identified solvents "
                x_pro_full = np.append(x_pro, x_pro[-n_input_pro])
                for (alias, prop) in zip(candidate_alias, candidate_props):
                    A.append([sol_index, alias, prop, x_pro])
                    prop = np.concatenate((prop, x_pro_full)).reshape(1, -1)
                    prop_std = prop_scaler_T1T2.transform(prop)
                    f_all = get_model_prediction(task, model_spec, get_Tensor(prop_std)).values()
                    F_ = np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F")
                    F.append(F_)

                from Aspen_d_ED_Solvent_T1T2 import run_Simulation
                _ = run_Simulation(candidate_alias, x_pro.reshape(1, -1), opt_path + "DistillRecoveryColumn.txt",
                                   solvent_Index=-1)
                KillAspen()
                time.sleep(5)

                for i in optimal_index:
                    optimal_indexes.append(i)
                # print([(candidate_alias[i] in ideal_alias, candidate_perfs[i]) for i in optimal_index])

            pred_columns = ["f_all_" + i for i in Models] + \
                           ["f_ave_" + i for i in Models] + \
                           ["f_std_" + i for i in Models]
            if column_index == "T1T2":
                columns = ["Index", "Alias", "Sol_prop", "Pro_var"] + pred_columns
            df = pd.DataFrame([ListValue2Str(a) + ListValue2Str(f) for (a, f) in zip(A, F)], columns=columns)
            df.to_csv(opt_path + "process_evaluation.csv")

            counter = collections.Counter(optimal_indexes)
            counter = collections.OrderedDict(counter.most_common())
            print("# Candidate occurrence:", dict(counter), flush=True)
            optimal_indexes = counter.keys()

            Solvent_list = [extra_alias[i] for i in optimal_indexes]
            Solevnt_prop = [extra_props[i] for i in optimal_indexes]
            Solevnt_std_prop = [extra_std_props[i] for i in optimal_indexes]
            Solvent_perf = [extra_perfs[i] for i in optimal_indexes]
            print(f"-> Solvents identified: {Solvent_list}")
            print(f"-> Weights considered & # candidates: {w != None}, {len(optimal_indexes)}", flush=True)
            print("-> Satisfied solvents:", np.sum([extra_alias[i] in ideal_alias for i in optimal_indexes]),
                  [str(i) + "-" + extra_alias[i] for i in optimal_indexes if extra_alias[i] in ideal_alias],
                  flush=True)

            # time.sleep(10)
            # exit()

            # optimization for each candidate
            puri_constr_dict = {
                "C4H6O2-N1": None,
                "C5H8O2-D1": [0.93, 0.98],
                "C4H7NO-E2": None,
                "C4H5N-1": [0.95, 0.98],
                "C4H8O2-D7": None,
                "C3H4O2-2": None,
                "C4H9NO-D0": None,
                "C4H6O3": [0.95, 0.98]
            }
            for (ind, (solvent, prop)) in enumerate(zip(Solvent_list, Solevnt_prop)):
                if solvent not in puri_constr_dict.keys(): continue
                print("\n#", ind, solvent, prop, perf)
                solvent_list = [solvent]
                algorithm = NSGA2(pop_size=100, sampling=sampling, crossover=crossover, mutation=mutation)
                problem_ = OptimProblem(task, column_index, error_constr_state, solvent_bound=prop,
                                        puri_constr=puri_constr_dict[solvent])

                " # Perform multi-objective optimization "
                res_ = minimize(problem_, algorithm, seed=params["seed"], save_history=True, verbose=True)
                np.save(opt_path + f"X_best_{str(ind)}_{solvent}.npy", res_.X)
                np.save(opt_path + f"F_best_{str(ind)}_{solvent}.npy", res_.F)
                pickle_save(opt_path + f"optimization_{str(ind)}_{solvent}.pickle", res_.history)
                print("-> Best solution found: %s" % res_.X, flush=True)
                print("-> Function value: %s" % res_.F, flush=True)

                if res_.X is not None:
                    Combine_loop = res_.X
                    if column_index == "T1":
                        Y = np.array([1 - res_.F[:, 0], perf_scaler_T1.inverse_transform(res_.F[:, 1])]).T
                        Combine_loop_ = prop_scaler_T1.transform(Combine_loop)
                        Combine_loop = Combine_loop[:, n_input_sol:]
                    elif column_index == "T2":
                        Y = np.array([1 - res_.F[:, 0], perf_scaler_T2.inverse_transform(res_.F[:, 1])]).T
                        Combine_loop_ = prop_scaler_T2.transform(Combine_loop)
                        Combine_loop = Combine_loop[:, n_input_sol:]
                    elif column_index == "T1T2":
                        Y = np.array([1 - res_.F[:, 0], perf_scaler_T1T2.inverse_transform(res_.F[:, 1])]).T
                        Combine_loop_14D = np.hstack((Combine_loop, Combine_loop[:, -n_input_pro].reshape(-1, 1)))
                        Combine_loop_ = prop_scaler_T1T2.transform(Combine_loop_14D)
                        Combine_loop = Combine_loop[:, n_input_sol:]

                    A, F = [], []
                    for x_std in Combine_loop_:
                        A.append([ind, solvent, prop])
                        x_std = x_std.reshape(1, -1)
                        if column_index != "T1T2":
                            f_all = get_model_prediction(task, model_spec, get_Tensor(x_std)).values()
                            F.append(
                                np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F"))
                        elif column_index == "T1T2":
                            f_all = get_model_prediction(task, model_spec, get_Tensor(x_std)).values()
                            F.append(
                                np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F"))

                    time.sleep(5)
                    if column_index == "T1T2":
                        from Aspen_d_ED_Solvent_T1T2 import run_Simulation
                    TimeStart = time.time()

                    txt_name = "DistillColumn.txt" if column_index == "T1" else "RecoveryColumn.txt" \
                        if column_index == "T2" else f"DistillRecoveryColumn_{ind}_{solvent}.txt"
                    _, _, Outputs = run_Simulation(solvent_list, Combine_loop, opt_path + txt_name)
                    KillAspen()
                    time.sleep(10)

                    TimeEnd = time.time()
                    print(f"Lunch & Simulation time:{TimeEnd - TimeStart: .2f} s for {len(Combine_loop)} runs",
                          flush=True)
                    pred_columns = ["f_all_" + i for i in Models] + \
                                   ["f_ave_" + i for i in Models] + \
                                   ["f_std_" + i for i in Models]
                    # if column_index == "T1":
                    #     columns = ["NStage_T1", "RR_T1", "TopPres_T1", "StoF"] + \
                    #               ["opt_DIST_C4H8_T1", "opt_RebDuty_T1"] + \
                    #               pred_columns + \
                    #               ["DIST_C4H8_T1", "RebDuty_T1", "hasIssue_T1", "hasERROR_T1"]
                    # elif column_index == "T2":
                    #     columns = ["NStage_T2", "RR_T2", "TopPres_T2", "StoF"] + \
                    #               ["opt_DIST_C4H6_T2", "opt_RebDuty_T2"] + \
                    #               pred_columns + \
                    #               ["DIST_C4H6_T2", "RebDuty_T2", "hasIssue_T2", "hasERROR_T2"]
                    if column_index == "T1T2":
                        columns = ["NStage_T1", "RR_T1", "TopPres_T1", "StoF", "NStage_T2", "RR_T2", "TopPres_T2"] + \
                                  ["opt_DIST_C4H8_T1", "opt_RebDuty_total"] + \
                                  pred_columns + \
                                  ["DIST_C4H8_T1", "DIST_C4H6_T1", "RebDuty_T1",
                                   "DIST_C4H6_T2", "DIST_SOL_T2", "RebDuty_T2",
                                   "hasIssue", "hasERROR", "CAPEX", "OPEX", "TAC"]
                    df = pd.DataFrame([ListValue2Str(c) + ListValue2Str(y) + ListValue2Str(f) + ListValue2Str(o)
                                       for c, y, f, o in zip(Combine_loop, Y, F, Outputs)], columns=columns)
                    if column_index == "T1T2":
                        df["RebDuty_T1"] = df["RebDuty_T1"].astype(float)
                        df["RebDuty_T2"] = df["RebDuty_T2"].astype(float)
                        df["RebDuty_total"] = df["RebDuty_T1"] + df["RebDuty_T2"]
                    df.to_csv(opt_path + f"process_evaluation_{ind}_{solvent}.csv")

            time.sleep(10)


if __name__ == "__main__":
    """
    -> IMPORTANT INFO: "model_linearity" should be set to False when using nonlinear models
    """
    " # 1: Solvent Design "
    # main("solvent", ("RebDuty_T1", "regression", "T1"))

    " # 2: Process Optimization "
    # main("process", ("RebDuty_T1", "regression", "T1"), error_constr=False)
    # main("process", ("RebDuty_T2", "regression", "T2"), error_constr=False)
    # main("process", ("RebDuty_T1T2", "regression", "T1T2"), error_constr=False)
    # main("process", ("RebDuty_T1T2", "regression", "T1T2"), error_constr=True)

    " # 3: Integrated Solvent and Process Design "
    # main("solvent_process", ("RebDuty_T1", "regression", "T1"), error_constr=False)
    # main("solvent_process", ("RebDuty_T2", "regression", "T2"), error_constr=False)
    main("solvent_process", ("RebDuty_T1T2", "regression", "T1T2"))
