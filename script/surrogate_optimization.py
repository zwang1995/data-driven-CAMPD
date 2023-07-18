# Created on 19 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Data-driven optimization for Solvent Design, Process Optimization, and Integrated Solvent and Process Design


import joblib
import collections
from basic_utils import *
from fnn_utils import *
from aspen_utils import *

import pandas as pd
from scipy.special import softmax
from pymoo.core.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from copy import deepcopy
from pymoo.config import Config

Config.show_compile_hint = False


def clean_screening_dataset(df_train, df_screen):
    train_sample_alias = df_train["alias"].tolist()
    df_screen = df_screen[df_screen["S_Aspen"] != "ERROR"]
    df_screen = df_screen[df_screen["CPMX_Aspen"] != "ERROR"]
    df_screen = df_screen[df_screen["MUMX_Aspen"] != "None"]
    df_screen = df_screen[~df_screen["alias"].isin(train_sample_alias)]
    df_cand = df_screen
    df_cand = df_cand.reset_index(drop=True)
    return df_cand


def get_model_arch(task, model_spec, print_state=False):
    params = get_param(task, model_spec)
    perf = model_spec[0]

    n_in, n_out = len(params["prop_label"]), len(params["perf_label"])
    n_layer, n_hid, act = eval(read_txt(params["best_hyper_file"])[0][0])
    NLayer_dict[perf] = n_layer
    AC_dict[perf] = get_AC(act)
    OutSig_dict[perf] = params["out_sig"] if perf not in ["hasERROR_T1", "hasERROR_T2"] else None

    if print_state:
        print(f"* Task: {perf} "
              f"| # hidden layers: {n_layer} "
              f"| # neurons: ({n_in}, {n_hid}, {n_out}) "
              f"| Activation function: {act}",
              flush=True)

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
            print("\tprediction:", perf_scaler_T1.inverse_transform(model(x_0).detach().numpy())[0][0], flush=True)
        elif perf == "RebDuty_T2":
            print("\tprediction:", perf_scaler_T2.inverse_transform(model(x_0).detach().numpy())[0][0], flush=True)
        else:
            print("\tprediction:", model(x_0).detach().numpy()[0][0], flush=True)
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
    def __init__(self, task, column_index, solvent_bound=None, puri_constr=None):
        self.task = task
        self.column_index = column_index
        self.solvent_bound = solvent_bound
        self.puri_constr = puri_constr

        def get_problem_setting(task, column_index):
            self.initial_state = True
            if task == "process":
                xl, xu = [40, 1, 3.5, 1, 8, 0.2, 3.5], [80, 10, 6, 8, 20, 2, 6]
                n_obj = 2
                n_constr = 3
            elif task == "solvent_process":
                if solvent_bound is not None:
                    solvent_l = solvent_bound * (1 - 1e-10)
                    solvent_u = solvent_bound * (1 + 1e-10)
                else:
                    solvent_l, solvent_u = [0.907091005, 1.32636452, 70.09104, 712.218755, 125.947982, 0.337070321], \
                                           [1.642181067, 5.944336533, 200.32136, 1182.04039, 412.977047, 2.47090885]
                xl, xu = np.concatenate((solvent_l, [40, 1, 3.5, 1, 8, 0.2, 3.5])), \
                         np.concatenate((solvent_u, [80, 10, 6, 8, 20, 2, 6]))
                n_obj = 2
                n_constr = 15 if solvent_bound is not None else 3
            return n_obj, n_constr, xl, xu

        n_obj, n_constr, xl, xu = get_problem_setting(task, column_index)
        print(f"* # OFs: {n_obj} "
              f"| # constraints: {n_constr} \n"
              f"  Lower bound: {[x for x in xl]} \n"
              f"  Upper bound: {[x for x in xu]}",
              flush=True)
        super().__init__(n_var=len(xl), n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        f_list, g_list = [], []
        X_ori = X

        # Input preprocessing
        if self.task == "process":
            X = np.concatenate((X, X[:, n_input_pro - 1].reshape(-1, 1)), axis=1)
            X = prop_scaler_T1T2.transform(X)
        elif self.task == "solvent_process":
            X_std = np.concatenate((X, X[:, n_input_sol + n_input_pro - 1].reshape(-1, 1)), axis=1)  # dimension 6+4+4
            X_std = prop_scaler_T1T2.transform(X_std)
            X = X_std

        # Obtain scaler properties
        if self.task == "process" or self.task == "solvent_process":
            f_duty_list, g_dutyL_list, g_dutyU_list = [], [], []
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
            if perf in Models: f = np.sum(f, axis=0)
            f = sigmoid(f) if out_sig else f
            return f

        for perf in Models:
            if self.task == "process":
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

            if self.task == "process":
                if perf == "DIST_C4H8_T1":
                    g_list.append(-np.average(f_, axis=0) + 0.9945)
                    if self.column_index == "T1":
                        f_list.append(1 - np.average(f_, axis=0))

                elif perf == "DIST_C4H6_T2":
                    g_list.append(-np.average(f_, axis=0) + 0.9945)
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
                    f_duty_list.append(mean_ + std_ * np.average(f_, axis=0))
                    g_dutyU_list.append(np.average(ori_f_, axis=0))


            elif self.task == "solvent_process":
                if self.puri_constr is None:
                    puri_a, puri_b = 0.995, 0.995
                else:
                    puri_a, puri_b = self.puri_constr

                if perf == "DIST_C4H8_T1":
                    g_list.append(-np.average(f_, axis=0) + puri_a)

                elif perf == "DIST_C4H6_T2":
                    g_list.append(-np.average(f_, axis=0) + puri_b)


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
                    g_dutyU_list.append(np.average(ori_f_, axis=0))

        mean_joint = perf_scaler_T1T2.mean_
        std_joint = perf_scaler_T1T2.scale_

        N_total = (X_ori[:, -2 * n_input_pro + 1] + X_ori[:, -n_input_pro + 1]) / 100
        f_list.append(N_total)
        f_list.append((np.sum(f_duty_list, axis=0) - mean_joint) / std_joint)
        g_list.append(np.sum(g_dutyU_list, axis=0) - 35000)

        out["F"] = np.column_stack(f_list)
        out["G"] = np.column_stack(g_list)

        if self.initial_state:
            print(f"* OF shape: {out['F'].shape} | Constraint shape: {out['G'].shape}")
            self.initial_state = False


def main(task, model_spec, nonlinear_state=True):
    # define global variables
    global n_input_sol, n_input_pro, Models
    global prop_scaler_T1, perf_scaler_T1, prop_scaler_T2, perf_scaler_T2, prop_scaler_T1T2, perf_scaler_T1T2
    _, FNN_task, column_index = model_spec

    # load settings
    if task == "process":
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
        params = get_param(task, ("RebDuty_T1", FNN_task, "T1T2"), nonlinear_state)
        n_input_sol = len(params["sol_label"])
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

    # set output path
    opt_path = create_directory(out_path + "optimization_" + column_index + "/")
    log_file = opt_path + "history.txt"
    with open(log_file, "w") as f:
        f.write("")
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)
    timestamp()

    # load model parameters
    global params_dict_list, NLayer_dict, AC_dict, OutSig_dict
    params_dict_list, NLayer_dict, AC_dict, OutSig_dict = {}, {}, {}, {}
    Models = ["DIST_C4H8_T1", "RebDuty_T1", "DIST_C4H6_T2", "RebDuty_T2"]
    for perf in Models:
        params_dict_list[perf] = []
        model_spec_ = (perf, "regression", column_index)
        params = get_param(task, model_spec_, nonlinear_state, False)
        model = get_model_arch(task, model_spec_, True)
        n_layer = NLayer_dict[perf]
        for i in range(5):
            params_dict = get_model_params(model, task, (perf, _, _), x_0, out_path, i, n_layer)
            params_dict_list[perf].append(deepcopy(params_dict))

    # optimization for CAMPD
    print("-> Optimization for CAMPD", flush=True)
    timestamp()

    if task == "process" or task == "solvent_process":
        if task == "process":
            mask = ["int"] + ["real"] * (n_input_pro - 1) + ["int"] + ["real"] * (n_input_pro - 2)
        elif task == "solvent_process":
            mask = ["real"] * n_input_sol + ["int"] + ["real"] * (n_input_pro - 1) + ["int"] + ["real"] * (
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
        if task == "process":
            algorithm = NSGA2(sampling=sampling, crossover=crossover, mutation=mutation)
        elif task == "solvent_process":
            algorithm = NSGA2(pop_size=1000, sampling=sampling, crossover=crossover, mutation=mutation)
    problem = OptimProblem(task, column_index)
    res = minimize(problem, algorithm, seed=params["seed"], save_history=True, verbose=True)
    timestamp()
    np.save(opt_path + "X_best.npy", res.X)
    np.save(opt_path + "F_best.npy", res.F)
    pickle_save(opt_path + "optimization.pickle", res.history)
    print("-> Optimization results", flush=True)
    print(f"* Optimal solutions: \n"
          f"{res.X}",
          flush=True)
    # print(res.X[:, :n_input_sol])
    # print(res.X[:, n_input_sol:])
    print(f"* OF values: \n"
          f"{res.F}",
          flush=True)

    # solvent mapping and process simulation
    if task == "process":
        Combine_loop = res.X
        Y = np.array([res.F[:, 0] * 100, perf_scaler_T1T2.inverse_transform(res.F[:, 1])]).T
        Combine_loop_8D = np.hstack((Combine_loop, Combine_loop[:, n_input_pro - 1].reshape(-1, 1)))
        Combine_loop_ = prop_scaler_T1T2.transform(Combine_loop_8D)

        F = []
        for x_std in Combine_loop_:
            x_std = x_std.reshape(1, -1)
            f_all = get_model_prediction(task, model_spec, get_Tensor(x_std)).values()
            F.append(np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F"))

        from Aspen_c_ED_T1T2 import run_Simulation
        TimeStart = time.time()

        total_run = len(Combine_loop)
        print(f"-> {total_run} simulations to be run", flush=True)
        run_Index = 0
        Outputs = []
        while run_Index < total_run - 1:
            txt_name = "DistillRecoveryColumn.txt"
            run_Index, output = run_Simulation(Combine_loop, opt_path + txt_name, run_Index)
            Outputs += output
            KillAspen()

        TimeEnd = time.time()
        print(f"* Simulation time:{TimeEnd - TimeStart: .2f} s for {len(Combine_loop)} runs", flush=True)
        pred_columns = ["f_all_" + i for i in Models] + \
                       ["f_ave_" + i for i in Models] + \
                       ["f_std_" + i for i in Models]
        columns = ["NStage_T1", "RR_T1", "TopPres_T1", "StoF", "NStage_T2", "RR_T2", "TopPres_T2"] + \
                  ["opt_N_total", "opt_RebDuty_total"] + \
                  pred_columns + \
                  ["DIST_C4H8_T1", "DIST_C4H6_T1", "RebDuty_T1",
                   "DIST_C4H6_T2", "DIST_SOL_T2", "RebDuty_T2",
                   "hasIssue", "hasERROR", "CAPEX", "OPEX", "TAC"]
        df = pd.DataFrame([ListValue2Str(c) + ListValue2Str(y) + ListValue2Str(f) + ListValue2Str(o)
                           for c, y, f, o in zip(Combine_loop, Y, F, Outputs)], columns=columns)
        df["RebDuty_T1"] = df["RebDuty_T1"].astype(float)
        df["RebDuty_T2"] = df["RebDuty_T2"].astype(float)
        df["RebDuty_total"] = df["RebDuty_T1"] + df["RebDuty_T2"]
        df.to_csv(opt_path + "process_evaluation.csv")

    elif task == "solvent_process":
        # solvent for matching
        df_train = pd.read_csv("../data/solvent/data_for_modeling.csv")
        df_screen = pd.read_csv("../data/solvent/data_all.csv")
        df_cand = clean_screening_dataset(df_train, df_screen)
        df_cand.to_csv(opt_path + "data_for_mapping.csv")

        prop_scaler_ = StandardScaler()
        prop_scaler_.mean_ = prop_scaler_T1.mean_[:n_input_sol]
        prop_scaler_.scale_ = prop_scaler_T1.scale_[:n_input_sol]

        print(f"\n* Candidate number: {len(df_cand)}", flush=True)
        extra_alias = df_cand["alias"].tolist()
        extra_props = np.array(df_cand[params["prop_label"][:n_input_sol]].values).astype(np.float)
        extra_std_props = prop_scaler_.transform(extra_props)

        # molecular mapping
        optimal_indexes = []
        A, F = [], []
        for (sol_index, x) in enumerate(res.X):
            x_sol, x_pro = x[:n_input_sol], x[n_input_sol:]
            x_sol_std = prop_scaler_.transform(x_sol.reshape(1, -1))
            # print(x_sol, x_pro)
            distance_dict = {}
            for (index, prop) in enumerate(extra_std_props):
                dist = Euclidean_distance(x_sol_std - prop)
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
        columns = ["Index", "Alias", "Sol_prop", "Pro_var"] + pred_columns
        df = pd.DataFrame([ListValue2Str(a) + ListValue2Str(f) for (a, f) in zip(A, F)], columns=columns)
        df.to_csv(opt_path + "process_evaluation.csv")

        counter = collections.Counter(optimal_indexes)
        counter = collections.OrderedDict(counter.most_common())
        print("* # Candidate occurrence:", dict(counter), flush=True)
        optimal_indexes = counter.keys()

        Solvent_list = [extra_alias[i] for i in optimal_indexes]
        Solevnt_prop = [extra_props[i] for i in optimal_indexes]
        # Solevnt_std_prop = [extra_std_props[i] for i in optimal_indexes]
        # Solvent_perf = [extra_perfs[i] for i in optimal_indexes]
        print(f"* Solvents identified: {Solvent_list}")
        print(f"* # candidates: {len(optimal_indexes)}", flush=True)

        " # Optimization for each candidate "
        print("-> Optimization for each candidate", flush=True)
        puri_constr_dict = {
            "C5H8O2-D1": [0.930, 0.985],
            "C4H7NO-E2": [0.965, 0.985],
            "C4H9NO-D0": [0.965, 0.985],
            "C4H6O2-N1": [0.965, 0.985],
            "C6H12O3-E2": [0.965, 0.985],
            "C5H8O3-D2": [0.965, 0.985],
            "C4H6O3": [0.960, 0.985],
            "C8H8O-D3": [0.965, 0.985],
            "C4H10N2": [0.965, 0.985]
        }
        for (ind, (solvent, prop)) in enumerate(zip(Solvent_list, Solevnt_prop)):
            if solvent not in puri_constr_dict.keys(): continue
            print(f"* Process optimization for solvent {solvent}")
            print(f"\n# {ind} solvent: {solvent}, property: {[x for x in prop]}")
            solvent_list = [solvent]
            algorithm = NSGA2(pop_size=100, sampling=sampling, crossover=crossover, mutation=mutation)
            problem_ = OptimProblem(task, column_index, solvent_bound=prop,
                                    puri_constr=puri_constr_dict[solvent])
            res_ = minimize(problem_, algorithm, seed=params["seed"], save_history=True, verbose=True)
            np.save(opt_path + f"X_best_{str(ind)}_{solvent}.npy", res_.X)
            np.save(opt_path + f"F_best_{str(ind)}_{solvent}.npy", res_.F)
            pickle_save(opt_path + f"optimization_{str(ind)}_{solvent}.pickle", res_.history)
            print(f"* Optimal solutions: \n"
                  f"{res_.X}",
                  flush=True)
            print(f"* OF values: \n"
                  f"{res_.F}",
                  flush=True)

            if res_.X is not None:
                Combine_loop = res_.X
                Y = np.array([res_.F[:, 0] * 100, perf_scaler_T1T2.inverse_transform(res_.F[:, 1])]).T
                Combine_loop_14D = np.hstack((Combine_loop, Combine_loop[:, -n_input_pro].reshape(-1, 1)))
                Combine_loop_ = prop_scaler_T1T2.transform(Combine_loop_14D)
                Combine_loop = Combine_loop[:, n_input_sol:]

                A, F = [], []
                for x_std in Combine_loop_:
                    x_std = x_std.reshape(1, -1)
                    f_all = get_model_prediction(task, model_spec, get_Tensor(x_std)).values()
                    F.append(np.array([[f, np.average(f), np.std(f)] for f in f_all], dtype=object).flatten("F"))

                print(f"* Process simulation for solvent {solvent}")
                TimeStart = time.time()
                from Aspen_d_ED_Solvent_T1T2 import run_Simulation
                txt_name = f"DistillRecoveryColumn_{ind}_{solvent}.txt"
                _, _, Outputs = run_Simulation(solvent_list, Combine_loop, opt_path + txt_name)
                KillAspen()
                TimeEnd = time.time()
                print(f"* Simulation time:{TimeEnd - TimeStart: .2f} s for {len(Combine_loop)} runs", flush=True)
                pred_columns = ["f_all_" + i for i in Models] + \
                               ["f_ave_" + i for i in Models] + \
                               ["f_std_" + i for i in Models]
                columns = ["NStage_T1", "RR_T1", "TopPres_T1", "StoF", "NStage_T2", "RR_T2", "TopPres_T2"] + \
                          ["opt_N_total", "opt_RebDuty_total"] + \
                          pred_columns + \
                          ["DIST_C4H8_T1", "DIST_C4H6_T1", "RebDuty_T1",
                           "DIST_C4H6_T2", "DIST_SOL_T2", "RebDuty_T2",
                           "hasIssue", "hasERROR", "CAPEX", "OPEX", "TAC"]
                df = pd.DataFrame([ListValue2Str(c) + ListValue2Str(y) + ListValue2Str(f) + ListValue2Str(o)
                                   for c, y, f, o in zip(Combine_loop, Y, F, Outputs)], columns=columns)
                df["RebDuty_T1"] = df["RebDuty_T1"].astype(float)
                df["RebDuty_T2"] = df["RebDuty_T2"].astype(float)
                df["RebDuty_total"] = df["RebDuty_T1"] + df["RebDuty_T2"]
                df.to_csv(opt_path + f"process_evaluation_{ind}_{solvent}.csv")
    timestamp()


if __name__ == "__main__":
    # 1: Process Optimization for Solvent NMP
    # main("process", ("RebDuty_T1T2", "regression", "T1T2"))

    # 2: CAMPD
    main("solvent_process", ("RebDuty_T1T2", "regression", "T1T2"))
