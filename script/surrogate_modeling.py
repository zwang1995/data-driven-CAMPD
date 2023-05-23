# Created at 05 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Data-driven modeling for Extractive Distillation Processes using Artificial Neural Networks

import json
import pandas as pd
from copy import deepcopy

from fnn_utils import *
from param_utils import *
from basic_utils import *
import torch

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import joblib, time


def clean_data(params, df):
    if params["task"] == "solvent":
        df = df[~df["hasERROR"]]
        df = df[df["DIST_C4H8_T1"] >= 0.5]
        df = df[df["DIST_C4H8_T1"] + df["DIST_C4H6_T1"] >= 0.99]

        df = df[df["RHO_Aspen"] <= 1200]
        df = df[df["MUMX_Aspen"] <= 3]

    elif params["task"] == "process":
        threshold = 0.5
        if params["column_index"] == "T1":
            df = df[~df["hasERROR_T1"]]
            df = df[df["DIST_C4H8_T1"] >= threshold]
            df = df.reset_index(drop=True)
            max_ind = np.where((df[params["prop_label"]] == [80, 10, 6, 8]).all(axis=1))
            max_duty = df.loc[max_ind]["RebDuty_T1"].values[0]
            ind = np.where(df["RebDuty_T1"] > max_duty)[0]
            df = df.drop(index=ind)

        elif params["column_index"] == "T2":
            df = df[~df["hasERROR_T2"]]
            df = df[df["DIST_C4H6_T2"] >= threshold]

    elif params["task"] == "solvent_process":
        threshold = 0.5
        if params["column_index"] == "T1":
            df = df[df["DIST_C4H8_T1"] > 0.1]
            df = df[df["RebDuty_T1"] > 0]
            df = df[~df["hasERROR_T1"]]
            df = df.reset_index(drop=True)
            INDEX = []
            for i in range(130):
                max_ind = np.where((df["Solvent_Index"] == i) &
                                   (df["NStage"] == 80) &
                                   (df["RR"] == 10) &
                                   (df["TopPres"] == 6) &
                                   (df["StoF"] == 8))
                if len(max_ind[0]) == 0:
                    max_ind = np.where((df["Solvent_Index"] == i) &
                                       (df["NStage"] == 75) &
                                       (df["RR"] == 10) &
                                       (df["TopPres"] == 6) &
                                       (df["StoF"] == 8))
                max_duty = df.loc[max_ind]["RebDuty_T1"].values[0]
                ind = np.where((df["Solvent_Index"] == i) & (df["RebDuty_T1"] > max_duty))[0]
                if len(ind) > 0: INDEX.extend(ind)

                if i in [85, 104, 110]:
                    add_ind = np.where((df["Solvent_Index"] == i) & (df["DIST_C4H8_T1"] < 0.5))[0]
                    INDEX.extend(add_ind)
                elif i == 123:
                    add_ind = np.where((df["Solvent_Index"] == i) & (df["DIST_C4H8_T1"] < 0.55))[0]
                    INDEX.extend(add_ind)
            INDEX = list(set(INDEX))
            df = df.drop(index=INDEX)
            df = df.reset_index(drop=True)

        elif params["column_index"] == "T2":
            df = df[~df["hasERROR_T2"]]
            # df = df[df["DIST_C4H6_T2"] >= threshold]
            # df = df[df["RebDuty_T2"] <= 5000]

    df = df.reset_index(drop=True)
    return df


def prepare_dataset(params, df):
    prop_label = params["prop_label"]
    sol_prop = df[prop_label].values
    perf_label = params["perf_label"]
    sol_perf = df[perf_label].values
    sol_name = df["alias"].values if params["task"] != "process" else None
    sol_smiles = df["CanonSMILES_PubChem"].values if params["task"] == "solvent" else None
    sol_num = len(df)

    items = []
    for i in range(sol_num):
        if params["task"] == "solvent":
            item = Molecule_reader(i, sol_name[i], sol_smiles[i], sol_prop[i], sol_perf[i])
        elif params["task"] == "process":
            item = Molecule_reader(i, None, None, sol_prop[i], sol_perf[i])
        elif params["task"] == "solvent_process":
            item = Molecule_reader(i, sol_name[i], None, sol_prop[i], sol_perf[i])
        items.append(item)
    return items


def split_data(params, items, test_size=None):
    # if params["task"] == "process":
    #     items_train, items_test = train_test_split(items, test_size=params["test_size"], random_state=params["seed"],
    #                                                stratify=[item.performance for item in items])
    if params["task"] != "solvent_process":
        if test_size:
            items_train, items_test = train_test_split(items, test_size=test_size, random_state=params["seed"])
        else:
            items_train, items_test = train_test_split(items, test_size=params["test_size"],
                                                       random_state=params["seed"])
    else:
        df_solvent = pd.read_csv(params["main_in_path"] + "solvent_list.csv")
        # df_solvent_train = df_solvent[df_solvent["set"] == "train"]
        # df_solvent_test = df_solvent[df_solvent["set"] == "test"]
        # solvent_list_train = df_solvent_train["alias"].values
        # solvent_list_test = df_solvent_test["alias"].values
        # print(solvent_list_train, solvent_list_test)
        global solvent_list_train
        solvent_list = df_solvent["alias"].values
        solvent_list_train, solvent_list_test = train_test_split(solvent_list, test_size=params["test_size"],
                                                                 random_state=params["seed"])
        # print(solvent_list_train)
        # print(solvent_list_train, solvent_list_test)

        items_train = [item for item in items if item.name in solvent_list_train]
        items_test = [item for item in items if item.name in solvent_list_test]
    if params["shuffle_state"]:
        shuffle_array(params, items_train)
        shuffle_array(params, items_test)
        # print([item.name for item in items_train])
        if params["task"] == "solvent_process":
            shuffle_array(params, solvent_list_train)
    return items_train, items_test


def get_prop_scaler(params, items_train=None):
    prop_scaler = None
    if params["prop_scaler_state"]:
        if params["task"] == "solvent" or params["task"] == "process":
            prop_train = np.array([item.property for item in items_train])
            prop_scaler = fit_scaler(prop_train)
            joblib.dump(prop_scaler, params["prop_scaler_file"])
            write_list_to_txt(params["prop_scaler_prop_file"], prop_scaler.mean_, "w")
            write_list_to_txt(params["prop_scaler_prop_file"], prop_scaler.scale_, "a")
        elif params["task"] == "solvent_process":
            prop_scaler_solvent = joblib.load(params["main_out_path"] + "prop_scaler_solvent.pkl")
            if params["column_index"] == "T1":
                prop_scaler_process = joblib.load(params["main_out_path"] + "prop_scaler_T1.pkl")
            elif params["column_index"] == "T2":
                prop_scaler_process = joblib.load(params["main_out_path"] + "prop_scaler_T2.pkl")
            prop_scaler = StandardScaler()
            prop_scaler.mean_ = np.concatenate((prop_scaler_solvent.mean_, prop_scaler_process.mean_))
            prop_scaler.scale_ = np.concatenate((prop_scaler_solvent.scale_, prop_scaler_process.scale_))
            joblib.dump(prop_scaler, params["prop_scaler_file"])
            write_list_to_txt(params["prop_scaler_prop_file"], prop_scaler.mean_, "w")
            write_list_to_txt(params["prop_scaler_prop_file"], prop_scaler.scale_, "a")
    return prop_scaler


def get_perf_scaler(params, items_train):
    perf_scaler = None
    if params["perf_scaler_state"]:
        perf_train = np.array([item.performance for item in items_train])
        perf_scaler = fit_scaler(perf_train)
        joblib.dump(perf_scaler, params["perf_scaler_file"])
        write_list_to_txt(params["perf_scaler_prop_file"], perf_scaler.mean_, "w")
        write_list_to_txt(params["perf_scaler_prop_file"], perf_scaler.scale_, "a")
    return perf_scaler


def get_X_and_Y(params, items):
    X = np.array([item.std_property for item in items])
    if params["perf_scaler_state"]:
        Y = np.array([item.std_performance for item in items])
    else:
        Y = np.array([item.performance for item in items])
    Y = Y.squeeze() if params["FNN_task"] == "classification" else Y
    return X, Y


def standardize_data(params, items):
    for item in items:
        if params["prop_scaler_state"]:
            item.std_property = params["prop_scaler"].transform([item.property])[0]
        if params["perf_scaler_state"]:
            item.std_performance = params["perf_scaler"].transform([item.performance])[0]
    return items


def initial_model(params, model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            assign_seed(params["seed"])
            weight_size = layer.weight.shape  # initial random seed
            layer.weight.data = torch.tensor(np.random.normal(0, 0.5, size=weight_size), dtype=torch.float32)


def evaluate_model(params, model, x, y):
    x = get_Tensor(x)
    y = get_Tensor(y)
    f = model(x)

    if params["FNN_task"] == "regression":
        y_np, f_np = y.detach().numpy(), f.detach().numpy()
        if params["perf_scaler_state"]:
            y_np, f_np = params["perf_scaler"].inverse_transform(y_np), \
                         params["perf_scaler"].inverse_transform(f_np)
        MAE = np.average(metrics.mean_absolute_error(y_np, f_np))
        MSE = metrics.mean_squared_error(y_np, f_np)
        RMSE = np.sqrt(MSE)
        R2 = metrics.r2_score(y_np, f_np)
        return MAE, MSE, RMSE, R2
    else:
        _, f = torch.max(f.data, 1)
        y_np, f_np = y.detach().numpy().astype(int), f.detach().numpy()
        out = metrics.confusion_matrix(y_np, f_np).ravel()
        return out


def training(params, model, x_train, y_train, x_test, y_test):
    x_train = get_Tensor(x_train)
    x_test = get_Tensor(x_test)
    y_train = get_Tensor(y_train) if params["FNN_task"] == "regression" else get_TensorLong(y_train)
    y_test = get_Tensor(y_test) if params["FNN_task"] == "regression" else get_TensorLong(y_test)

    criterion = RMSELoss if params["FNN_task"] == "regression" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=params["scheduler_factor"],
                                                           patience=params["scheduler_patience"],
                                                           min_lr=params["min_lr"])
    early_stopping = EarlyStopping()

    for i in range(params["epoch"]):
        model.train()

        f_train = model(x_train)
        loss = criterion(f_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        if params["FNN_task"] == "regression":
            _, _, RMSE_2, R2_2 = evaluate_model(params, model, x_test, y_test)
            early_stopping(RMSE_2)
            if params["scheduler_state"]:
                scheduler.step(RMSE_2)
            if early_stopping.update:
                R2_2_out = R2_2
            if early_stopping.early_stop:
                break
        else:
            f_test = model(x_test)
            loss_test = float(criterion(f_test, y_test).detach().numpy())
            early_stopping(loss_test)
            if params["scheduler_state"]:
                scheduler.step(loss_test)
            if early_stopping.update:
                cm_out = evaluate_model(params, model, x_test, y_test)
            if early_stopping.early_stop:
                break

    if params["FNN_task"] == "regression":
        return model, i + 1, R2_2_out, early_stopping.best_loss
    else:
        return model, i + 1, cm_out, early_stopping.best_loss


def optimize_hyper(params, model, items_train, save_model=False):
    # print(model.state_dict()["l1.weight"].dtype)
    # initial_model(model)
    # print(model.state_dict()["l1.weight"])

    def get_X_and_Y_for_S_P(i):
        """ only apply for task 'solvent_process' """
        kf = KFold(n_splits=5)
        for j, (solvent_train, solvent_test) in enumerate(kf.split(solvent_list_train)):
            if i == j:
                cv_solvent_list_train = solvent_list_train[solvent_train]
                cv_solvent_list_test = solvent_list_train[solvent_test]
        cv_items_train = [item for item in items_train if item.name in cv_solvent_list_train]
        cv_items_test = [item for item in items_train if item.name in cv_solvent_list_test]
        x_train_cv, y_train_cv = get_X_and_Y(params, cv_items_train)
        x_test_cv, y_test_cv = get_X_and_Y(params, cv_items_test)
        return x_train_cv, y_train_cv, x_test_cv, y_test_cv

    X_train, Y_train = get_X_and_Y(params, items_train)

    kf = KFold(n_splits=5)
    model_params, epochs, Is, scores = [], [], [], []
    for i, (cv_index_train, cv_index_test) in enumerate(kf.split(X_train)):
        if params["model_initial_state"]:
            initial_model(params, model)
        # print(cv_index_train, cv_index_test)
        if params["task"] != "solvent_process":
            x_train_cv = X_train[cv_index_train]
            y_train_cv = Y_train[cv_index_train]
            x_test_cv = X_train[cv_index_test]
            y_test_cv = Y_train[cv_index_test]
        else:
            x_train_cv, y_train_cv, x_test_cv, y_test_cv = get_X_and_Y_for_S_P(i)

        if params["FNN_task"] == "regression":
            model, epoch, R2, score = training(params, model, x_train_cv, y_train_cv, x_test_cv, y_test_cv)
            Is.append(R2)
        else:
            model, epoch, cm, score = training(params, model, x_train_cv, y_train_cv, x_test_cv, y_test_cv)
            try:
                tn, fp, fn, tp = cm
                accu, prec, reca = (tn + tp) / (tn + fp + fn + tp), tp / (tp + fp), tp / (tp + fn)
            except:
                accu, prec, reca = 0., 0., 0.
            Is.append(np.array([accu, prec, reca]))
        model_params.append(deepcopy(model).state_dict())
        epochs.append(epoch)
        scores.append(score)

    return epochs, Is, scores, model_params


def save_predictions(params, final_model, file):
    if not params["perf_scaler_state"]:
        write_csv(file, ["no", "name", "smiles", "raw_Feature", "tran_Feature",
                         "Target", "pred_Target", "set"])
    else:
        write_csv(file, ["no", "name", "smiles", "raw_Feature", "tran_Feature",
                         "Target", "pred_Target", "std_Target", "pred_std_Target", "set"])

    def save_for_items(items, set):
        for item in items:
            x = get_Tensor(item.std_property)
            y = item.performance[0]
            f = final_model(x)
            f_np = f.detach().numpy()
            f_np = f_np[0] if params["FNN_task"] == "regression" else np.argmax(f_np)

            if not params["perf_scaler_state"]:
                write_csv(file, [item.no, item.name, item.smiles,
                                 item.property, item.std_property, y, f_np, set], "a")

            else:
                f_std = f_np
                y_std = item.std_performance[0]

                f_np = f.detach().numpy()
                f_np = params["perf_scaler"].inverse_transform(f_np)[0]

                write_csv(file, [item.no, item.name, item.smiles,
                                 item.property, item.std_property, y, f_np, y_std, f_std, set], "a")

    save_for_items(params["items_train"], "train")
    save_for_items(params["items_test"], "test")


def main(task, model_spec=None, nonlinear_state=True):
    params = get_param(task, model_spec, nonlinear_state, modeling_state=True)
    if params["FNN_task"] == "classification":
        if (params["column_index"] == "T1") and (not params["class_train_state_T1"]): return
        if (params["column_index"] == "T2") and (not params["class_train_state_T2"]): return
    with open(params["params_file"], "w") as fp:
        json.dump(params, fp, indent=4)
    print(f"-> Task & Perf & Nonlinear_state: {task}, {model_spec}, {nonlinear_state}", flush=True)
    print(f"-> Data used for modeling {params['data_file']}", flush=True)
    time_start = time.time()

    " # Random Seed "
    assign_seed(params["seed"])

    " # Preparation "
    df = pd.read_csv(params["data_file"])
    # prop_scaler trained based on all samples for task "process"
    items = prepare_dataset(params, df)
    print(f"-> Original dataset size: {len(items)}", flush=True)
    if params["task"] == "process" or params["task"] == "solvent_process":
        params["prop_scaler"] = get_prop_scaler(params, items)
    # data preprocessing
    if params["FNN_task"] == "regression":
        df = clean_data(params, df)
        print(f"-> Cleaned dataset size: {len(df)}", flush=True)
    elif params["FNN_task"] == "classification":
        df.loc[:, params["perf_label"][0]].replace({False: 1, True: 0}, inplace=True)  # Positive = correct simu
    # data preparation
    df.to_csv(params["used_data_file"])
    items = prepare_dataset(params, df)
    items_train, items_test = split_data(params, items)
    print(f"-> Training & Test size: {len(items_train)}, {len(items_test)}", flush=True)
    if params["task"] == "solvent":
        params["prop_scaler"] = get_prop_scaler(params, items_train)
    params["perf_scaler"] = get_perf_scaler(params, items_train)
    items = standardize_data(params, items)
    params["items_train"], params["items_test"] = items_train, items_test = split_data(params, items)
    n_in, n_out = len(params["prop_label"]), len(params["perf_label"])

    " # Hyperparameter Optimization "
    hyper_combs = get_hyper_comb(params)
    hyper_loss, hyper_model = {}, {}
    write_csv(params["hyper_opt_his"],
              ["hyper", "ave_epoch", "epoch", "ave_indicator", "indicator", "ave_score", "socre"], "w")
    for hyper_comb in hyper_combs:
        n_layer, n_hid, act = hyper_comb
        model = FNN_model(params, n_layer, (n_in, n_hid, n_out), act)
        epochs, Is, scores, model_params = optimize_hyper(params, model, items_train, save_model=False)
        hyper_loss[hyper_comb] = np.average(scores)
        hyper_model[hyper_comb] = model_params
        print([hyper_comb, np.average(epochs), np.average(Is, axis=0), np.average(scores), scores], flush=True)
        write_csv(params["hyper_opt_his"], [hyper_comb,
                                            np.average(epochs), epochs,
                                            np.average(Is, axis=0), Is,
                                            np.average(scores), scores], "a")

    " # Model Training "
    opt_hyper = min(hyper_loss, key=hyper_loss.get)
    print(f"-> Optimal hypers & loss: {opt_hyper}, {hyper_loss[opt_hyper]}", flush=True)
    write_list_to_txt(params["best_hyper_file"], [opt_hyper, hyper_loss[opt_hyper]])
    opt_n_layer, opt_n_hid, opt_act = opt_hyper
    opt_model = FNN_model(params, opt_n_layer, (n_in, opt_n_hid, n_out), opt_act)
    print(f"-> Optimal model structure: {opt_model}", flush=True)
    print("-> Trainable parameters:", sum(p.numel() for p in opt_model.parameters() if p.requires_grad), flush=True)
    save_model_structure(params, str(opt_model))
    opt_params = hyper_model[opt_hyper]
    X_train, y_train = get_X_and_Y(params, items_train)
    X_test, y_test = get_X_and_Y(params, items_test)

    print(f"-> Model evaluation ...", flush=True)
    columns = ["MAE_train", "MSE_train", "RMSE_train", "R2_train",
               "MAE_test", "MSE_test", "RMSE_test", "R2_test"] if params["FNN_task"] == "regression" \
        else ["TN_train", "FP_train", "FN_train", "TP_train",
              "TN_test", "FP_test", "FN_test", "TP_test"]
    write_csv(params["final_model_his"], columns)
    model_index = 0
    for model_param in opt_params:
        prefix = "# Model_" + str(model_index) + ":"
        opt_model.load_state_dict(model_param)
        x_0 = get_x_0(params)
        x_0 = get_Tensor(params["prop_scaler"].transform(x_0))
        print(prefix, params["perf_scaler"].inverse_transform(opt_model(x_0).detach().numpy())[0], flush=True) \
            if params["perf_scaler_state"] else print(prefix, opt_model(x_0).detach().numpy()[0], flush=True)
        torch.save(opt_model.state_dict(), "".join([params["out_path"], "model_", str(model_index), ".pt"]))
        # print(opt_model.state_dict())
        MAE_1, MSE_1, RMSE_1, R2_1 = evaluate_model(params, opt_model, X_train, y_train)
        MAE_2, MSE_2, RMSE_2, R2_2 = evaluate_model(params, opt_model, X_test, y_test)
        results = [MAE_1, MSE_1, RMSE_1, R2_1, MAE_2, MSE_2, RMSE_2, R2_2]
        print(f"Train & Test results: {results}", flush=True)
        write_csv(params["final_model_his"], results, "a")
        result_file = "".join([params["out_path"], "model_pred_", str(model_index), ".csv"])
        save_predictions(params, opt_model, result_file)
        model_index += 1

    write_string_to_txt(params["logging_file"], "")
    time_end = time.time()
    print("-> Time cost:", time_end - time_start, flush=True)
    print("", flush=True)


if __name__ == "__main__":
    """ # Problem 1: Solvent Design """
    # main("solvent", ("DIST_C4H8_T1", "regression", "T1"), True)
    # main("solvent", ("RebDuty_T1", "regression", "T1"), True)

    """ # Problem 2: Process Optimization """
    # " T1 "
    # main("process", ("DIST_C4H8_T1", "regression", "T1"), True)
    # time.sleep(100)
    # main("process", ("RebDuty_T1", "regression", "T1"), True)
    # time.sleep(100)
    # main("process", ("hasERROR_T1", "classification", "T1"), True)
    # " T2 "
    # main("process", ("DIST_C4H6_T2", "regression", "T2"), True)
    # time.sleep(100)
    # main("process", ("RebDuty_T2", "regression", "T2"), True)
    # time.sleep(100)
    # main("process", ("hasERROR_T2", "classification", "T2"), True)

    """ # Problem 3: Integrated Solvent and Process Design """
    # " T1 "
    main("solvent_process", ("DIST_C4H8_T1", "regression", "T1"), True)
    main("solvent_process", ("RebDuty_T1", "regression", "T1"), True)
    # main("solvent_process", ("hasERROR_T1", "classification", "T1"), True)
    # " T2 "
    main("solvent_process", ("DIST_C4H6_T2", "regression", "T2"), True)
    main("solvent_process", ("RebDuty_T2", "regression", "T2"), True)
    # main("solvent_process", ("hasERROR_T2", "classification", "T2"), True)
