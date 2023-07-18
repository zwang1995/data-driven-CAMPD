# Created on 08 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Settings for data-driven modeling and optimization

import itertools
from basic_utils import create_directory, SequenceWithEndPoint


def get_param(task=None, model_spec=("RebDuty_T1", "regression", "T1"), nonlinear_state=True, modeling_state=False):
    specify_perf, FNN_task, column_index = model_spec
    params = {
        "task": task,  # "solvent" or "process"
        "seed": 42,
        "test_size": 0.2,

        # preprocessing
        "prop_scaler_state": True,  # apply to all perfs
        "shuffle_state": True,  # shuffle after split data

        # model
        "model_nonlinear": nonlinear_state,  # linear model (False) or nonlinear model (True)
        "perf_scaler_state": True if (specify_perf is not None) and ("RebDuty" in specify_perf) else False,
        "model_initial_state": True,  # initial model parameters once calling the model

        # optimizer
        "epoch": 10000,  # maximal training epoch
        "lr": 0.02 if task == "process" else 0.005,  # initial learning rate
        "earlystop_patience": 10,
        "weight_decay": 0,  # 1e-6,
        "scheduler_state": True if "process" in task else False,
        "min_lr": 0.0001,
        "scheduler_factor": 0.9,
        "scheduler_patience": 5,
    }

    if task is not None:
        params["main_in_path"] = "../data/" + task + "/"

        # * Task 1: Solvent Design
        if task == "solvent":
            params["FNN_task"] = "regression"
            # FNN structure setting: # hidden layer, whether the output layer uses sigmoid
            if specify_perf == "DIST_C4H8_T1":
                params["out_sig"] = True if params["model_nonlinear"] else False
            elif specify_perf == "RebDuty_T1":
                params["out_sig"] = False if params["model_nonlinear"] else False

            params["data_file"] = params["main_in_path"] + "data_for_modeling.csv"
            params["sol_label"] = ["S_Aspen", "MW_Aspen", "RHO_Aspen", "CPMX_Aspen", "MUMX_Aspen"]
            params["prop_label"] = params["sol_label"]
            # params["prop_label"] = ["S_Aspen", "Alpha_Aspen", "CPMX_Aspen"]
            params["perf_label"] = ["DIST_C4H8_T1", "RebDuty_T1"] if specify_perf is None else [specify_perf]

        # * Task 2: Process Design
        elif task == "process":
            params["FNN_task"] = FNN_task  # "regression", "classification"
            params["column_index"] = column_index
            params["prop_label"] = [f"{label}_{column_index}" for label in ["NStage", "RR", "TopPres", "StoF"]]
            if column_index == "T1":
                params["data_file"] = params["main_in_path"] + "DistillColumn_NMP.csv"
            elif column_index == "T2":
                params["data_file"] = params["main_in_path"] + "RecoveryColumn_NMP.csv"

            if params["FNN_task"] == "classification": params["lr"] = 0.02

            if params["FNN_task"] == "regression":
                # FNN structure setting: # hidden layer, whether the output layer uses sigmoid
                if specify_perf == "DIST_C4H8_T1" or specify_perf == "DIST_C4H6_T2":
                    params["out_sig"] = True if params["model_nonlinear"] else False
                elif (specify_perf == "RebDuty_T1") or (specify_perf == "RebDuty_T2"):
                    params["out_sig"] = False if params["model_nonlinear"] else False
                if specify_perf is None:
                    if column_index == "T1":
                        params["perf_label"] = ["DIST_C4H8_T1", "RebDuty_T1"]
                    elif column_index == "T2":
                        params["perf_label"] = ["DIST_C4H6_T2", "RebDuty_T2"]
                else:
                    params["perf_label"] = [specify_perf]
            elif params["FNN_task"] == "classification":
                if specify_perf is None:
                    if column_index == "T1":
                        params["perf_label"] = ["hasERROR_T1"]
                    elif column_index == "T2":
                        params["perf_label"] = ["hasERROR_T2"]
                else:
                    params["perf_label"] = [specify_perf]
                    params["class_train_state_T1"] = False
                    params["class_train_state_T2"] = True

        # * Task 3: Integrated Solvent and Process Design
        elif task == "solvent_process":
            params["FNN_task"] = FNN_task  # "regression", "classification"
            params["column_index"] = column_index
            if column_index == "T1":
                params["data_file"] = params["main_in_path"] + "DistillColumn_solvent.csv"
            elif column_index == "T2":
                params["data_file"] = params["main_in_path"] + "RecoveryColumn_solvent.csv"
            params["sol_label"] = ["S_Aspen", "Alpha_Aspen", "MW_Aspen", "RHO_Aspen", "CPMX_Aspen", "MUMX_Aspen"]
            params["pro_label"] = ["NStage", "RR", "TopPres", "StoF"]
            params["prop_label"] = params["sol_label"] + params["pro_label"]

            if params["FNN_task"] == "regression":
                # FNN structure setting: # hidden layer, whether the output layer uses sigmoid
                if specify_perf == "DIST_C4H8_T1" or specify_perf == "DIST_C4H6_T2":
                    params["out_sig"] = True if params["model_nonlinear"] else False
                elif (specify_perf == "RebDuty_T1") or (specify_perf == "RebDuty_T2"):
                    params["out_sig"] = False if params["model_nonlinear"] else False
                if specify_perf is None:
                    if column_index == "T1":
                        params["perf_label"] = ["DIST_C4H8_T1", "RebDuty_T1"]
                    elif column_index == "T2":
                        params["perf_label"] = ["DIST_C4H6_T2", "RebDuty_T2"]
                else:
                    params["perf_label"] = [specify_perf]
            elif params["FNN_task"] == "classification":
                if specify_perf is None:
                    if column_index == "T1":
                        params["perf_label"] = ["hasERROR_T1"]
                    elif column_index == "T2":
                        params["perf_label"] = ["hasERROR_T2"]
                else:
                    params["perf_label"] = [specify_perf]
                    params["class_train_state_T1"] = False
                    params["class_train_state_T2"] = False

        # Create folder
        params["main_out_path"] = "../model/" + task + "/"
        params["viz_path"] = create_directory("../viz/" + task + "/")

        # Setting-related files
        out_path = params["main_out_path"] + params["perf_label"][0] + "/"
        if modeling_state:
            params["out_path"] = create_directory(out_path)
        else:
            params["out_path"] = out_path
        params["used_data_file"] = "".join([params["out_path"], "used_data.csv"])
        params["params_file"] = "".join([params["out_path"], "params.json"])
        params["model_stru_file"] = "".join([params["out_path"], "model_stru.txt"])
        params["log_file"] = "".join([params["out_path"], "history.log"])

        # Model-related files
        params["prop_scaler_file"] = "".join([params["out_path"], "prop_scaler.pkl"])
        params["perf_scaler_file"] = "".join([params["out_path"], "perf_scaler.pkl"])
        params["prop_scaler_prop_file"] = "".join([params["out_path"], "prop_scaler.txt"])
        params["perf_scaler_prop_file"] = "".join([params["out_path"], "perf_scaler.txt"])
        params["hyper_opt_his"] = "".join([params["out_path"], "hyper_opt_his.csv"])
        params["best_hyper_file"] = "".join([params["out_path"], "best_hyper.txt"])
        params["final_model_his"] = "".join([params["out_path"], "final_model_his.csv"])
        params["model_pred_file"] = "".join([params["out_path"], "model_pred.csv"])
    return params


def get_hyper_comb(params):
    HLayer_list = SequenceWithEndPoint(1, 2, 1)
    if params["task"] == "solvent":
        NHid_list = SequenceWithEndPoint(1, 4, 1)  # limit model complexity
    elif params["task"] == "process":
        NHid_list = SequenceWithEndPoint(1, 16, 1)
    elif params["task"] == "solvent_process":
        NHid_list = SequenceWithEndPoint(8, 24, 1)
    Act_list = ["elu", "sigmoid", "softplus", "tanh"] if params["model_nonlinear"] else ["identity"]
    Combine_loop = list(itertools.product(HLayer_list, NHid_list, Act_list))
    return Combine_loop
