# Created on 25 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Utils used for visualization

import itertools

import scipy.stats as stats
from param_utils import *
from basic_utils import *
from surrogate_modeling import clean_data
from surrogate_optimization import OptimProblem

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import font_manager

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# print(plt.rcParams)

# For HPC use
# ttf_path = "arial.ttf"
# font_manager.fontManager.addfont(ttf_path)
# End

# plt.rcParams["font.size"] = "9"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.figsize"] = (3.3, 3)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["xtick.minor.width"] = 0.6
plt.rcParams["legend.handletextpad"] = 0.3
plt.rcParams["figure.dpi"] = 600.0
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True
plt.rcParams["agg.path.chunksize"] = 10000

mode = "doc"
print(f"Mode: {mode}")

if mode == "doc":
    plt.rcParams["font.size"] = "9"
    label_size = 9.5
    text_size = 8.5
    legend_size = 8
    dot_size = 20
elif mode == "ppt":
    plt.rcParams["font.size"] = "11"
    label_size = 11.5
    text_size = 10
    legend_size = 9.5
    dot_size = 30


def extend_xylim(low, up):
    percent = 0.05
    range = up - low
    return [low - percent * range, up + percent * range]


" -------------------------------------------------------------------------------------------------------------------- "


def correlation_analysis(params, df, path):
    plt.clf()
    if params["task"] == "solvent":
        xlabel_dict = {"S_Aspen": "Selectivity",
                       "MW_Aspen": "Molecular weight (g/mol)",
                       "RHO_Aspen": "Density (kg/m$^3$)",
                       "CPMX_Aspen": "Heat capacity (kJ/kmol K)",
                       "MUMX_Aspen": "Viscosity (cP)"}
        ylabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                       "RebDuty_T1": "Reboiler heat duty (MW)"}
        xlim_dict = {"S_Aspen": extend_xylim(0.8, 1.8),
                     "MW_Aspen": extend_xylim(60, 180),
                     "RHO_Aspen": extend_xylim(600, 1800),
                     "CPMX_Aspen": extend_xylim(100, 400),
                     "MUMX_Aspen": extend_xylim(0, 10)}
        ylim_dict = {"DIST_C4H8_T1": extend_xylim(0, 1),
                     "RebDuty_T1": extend_xylim(0, 12000 / 10 ** 3)}
    elif params["task"] == "process":
        xlabel_dict = {"NStage": "Total number of stages",
                       "RR": "Reflux ratio",
                       "TopPres": "Operating pressure",
                       "StoF": "Solvent-to-feed ratio"}
        if params["column_index"] == "T1":
            ylabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                           "RebDuty_T1": "Reboiler heat duty (MW)"}
        elif params["column_index"] == "T2":
            ylabel_dict = {"DIST_C4H6_T2": "C$_4$H$_6$ purity",
                           "RebDuty_T2": "Reboiler heat duty (MW)"}
        xlim_dict = {"S_Aspen": extend_xylim(0.8, 1.8),
                     "MW_Aspen": extend_xylim(60, 180),
                     "RHO_Aspen": extend_xylim(600, 1800),
                     "CPMX_Aspen": extend_xylim(100, 400),
                     "MUMX_Aspen": extend_xylim(0, 10)}
        ylim_dict = {"DIST_C4H8": extend_xylim(0, 1),
                     "RebDuty": extend_xylim(0, 12000 / 10 ** 3)}
    elif params["task"] == "solvent_process":
        xlabel_dict = {"S_Aspen": "Selectivity",
                       "Alpha_Aspen": "Relative volatility",
                       "MW_Aspen": "Molecular weight (g/mol)",
                       "RHO_Aspen": "Density (kg/m$^3$)",
                       "CPMX_Aspen": "Heat capacity (kJ/kmol K)",
                       "MUMX_Aspen": "Viscosity (cP)",
                       "NStage": "Total number of stages",
                       "RR": "Reflux ratio",
                       "TopPres": "Operating pressure",
                       "StoF": "Solvent-to-feed ratio"}
        if params["column_index"] == "T1":
            ylabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                           "RebDuty_T1": "Reboiler heat duty (MW)"}
        elif params["column_index"] == "T2":
            ylabel_dict = {"DIST_C4H6_T2": "C$_4$H$_6$ purity",
                           "RebDuty_T2": "Reboiler heat duty (MW)"}

    if params["task"] == "solvent":
        his_file = path + "his.txt"
    else:
        his_file = path + params["column_index"] + "_his.txt"
    write_string_to_txt(his_file, f"# Data: {len(df)}", "w")
    print(f"# of data points: {len(df)}")
    for i, x in enumerate(list(xlabel_dict.keys())):
        for y in list(ylabel_dict.keys()):
            plt.clf()
            # print(df[x])
            xy = np.vstack([df[x], df[y]])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            df_x, df_y, df_z = df[x][idx], df[x][idx], z[idx]

            if "RebDuty" in y:
                plt.scatter(df_x, df_y / 10 ** 3, c=df_z, s=dot_size)
            else:
                plt.scatter(df_x, df_y, c=df_z, s=dot_size)

            # plt.xlim(xlim_dict[x])
            # plt.ylim(ylim_dict[y])
            plt.xlabel(xlabel_dict[x], size=label_size)
            plt.ylabel(ylabel_dict[y], size=label_size)
            plt.savefig(path + str(i) + "_" + x + "&" + y)
            r, p = stats.pearsonr(df[x], df[y])
            print((y, x, r, p))
            write_string_to_txt(his_file, f"Pair {y} & {x}: r={r}, p={p}", "a")


" ---------------------------------------------------------- "


def Figure_0a():
    """ Analyze the linear correlation between molecular property x
        and process performance y and visualization for 126 candidates """
    params = get_param("solvent", ("RebDuty", "regression", "T1"))
    df = pd.read_csv(params["data_file"])
    sub_viz_path = params["viz_path"] + mode + "/corr_ana/"
    create_directory(sub_viz_path)
    df = clean_data(params, df)
    correlation_analysis(params, df, sub_viz_path)
    plt.close()


def Figure_0b(column_index):
    """ Analyze the linear correlation between molecular property x
        and process performance y and visualization for 126 candidates """
    params = get_param("process", ("RebDuty_T1", "regression", column_index))
    df = pd.read_csv(params["data_file"])
    print(len(df))
    sub_viz_path = params["viz_path"] + mode + "/corr_ana/"
    create_directory(sub_viz_path)
    df = clean_data(params, df)
    print(len(df))
    correlation_analysis(params, df, sub_viz_path)
    plt.close()


def Figure_0c(column_index):
    """ Analyze the linear correlation between molecular property x
        and process performance y and visualization for 126 candidates """
    params = get_param("solvent_process", ("RebDuty_T1", "regression", column_index))
    df = pd.read_csv(params["data_file"])
    print(len(df))
    sub_viz_path = params["viz_path"] + mode + "/corr_ana/"
    create_directory(sub_viz_path)
    df = clean_data(params, df)
    print(len(df))
    correlation_analysis(params, df, sub_viz_path)
    plt.close()


" #################################################################################################################### "


def parity_a(params, res_main_path, path, model_spec, if_merge=False):
    plt.clf()
    y_scale = 1000
    if params["task"] == "solvent":
        xlabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                       "RebDuty_T1": "Reboiler heat duty (MW)"}
        ylabel_dict = {"DIST_C4H8_T1": "Predicted purity",
                       "RebDuty_T1": "Predicted heat duty (MW)"}
        xylim_dict = {"DIST_C4H8_T1": extend_xylim(0.48, 1.02),  # extend_xylim(0.48, 1)
                      "RebDuty_T1": extend_xylim(8000 / y_scale, 52000 / y_scale)}
        diagonal_dict = {"DIST_C4H8_T1": (0, 2), "RebDuty_T1": (0 / y_scale, 60000 / y_scale)}
        unit_dict = {"DIST_C4H8_T1": "",
                     "RebDuty_T1": " MW"}
    elif params["task"] == "process":
        xlabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                       "DIST_C4H6_T2": "C$_4$H$_6$ purity",
                       "RebDuty_T1": "EDC heat duty (MW)",
                       "RebDuty_T2": "SRC heat duty (MW)"}
        ylabel_dict = {"DIST_C4H8_T1": "Predicted C$_4$H$_8$ purity",
                       "DIST_C4H6_T2": "Predicted C$_4$H$_6$ purity",
                       "RebDuty_T1": "Predicted EDC heat duty (MW)",
                       "RebDuty_T2": "Predicted SRC heat duty (MW)"}
        xylim_dict = {"DIST_C4H8_T1": extend_xylim(0.68, 1),  # extend_xylim(0.48, 1)
                      "DIST_C4H6_T2": extend_xylim(0.68, 1),
                      "RebDuty_T1": extend_xylim(0 / y_scale, 80000 / y_scale),
                      "RebDuty_T2": extend_xylim(5000 / y_scale, 25000 / y_scale)}
        diagonal_dict = {"DIST_C4H8_T1": (0, 2), "RebDuty_T1": (-10000 / y_scale, 90000 / y_scale),
                         "DIST_C4H6_T2": (0, 2), "RebDuty_T2": (-10000 / y_scale, 40000 / y_scale)}
        unit_dict = {"DIST_C4H8_T1": "",
                     "DIST_C4H6_T2": "",
                     "RebDuty_T1": " MW",
                     "RebDuty_T2": " MW"}
        xticks_dict = {"DIST_C4H8_T1": np.arange(0.7, 1.0, 0.1),
                       "DIST_C4H6_T2": np.arange(0.7, 1.0, 0.1),
                       "RebDuty_T1": np.arange(0, 90, 20),
                       "RebDuty_T2": np.arange(5, 30, 5)}

    l, perf = model_spec
    res_path = res_main_path + "/" + perf + "/"
    score_file = res_path + "final_model_his.csv"
    df_score = pd.read_csv(score_file)
    MAE_train, R2_train = df_score["MAE_train"].values, df_score["R2_train"].values
    MAE_test, R2_test = df_score["MAE_test"].values, df_score["R2_test"].values

    MAE_train_mean, MAE_train_std = np.average(MAE_train), np.std(MAE_train)
    R2_train_mean, R2_train_std = np.average(R2_train), np.std(R2_train)
    MAE_test_mean, MAE_test_std = np.average(MAE_test), np.std(MAE_test)
    R2_test_mean, R2_test_std = np.average(R2_test), np.std(R2_test)

    xy_low, xy_up = xylim_dict[perf]
    x_low, x_up = diagonal_dict[perf]

    if mode == "doc":
        x_offset = 0.1
        x_text_MAE_train, y_text_MAE_train = xy_low + (0.05 + x_offset) * (xy_up - xy_low), \
                                             xy_low + 0.7 * (xy_up - xy_low)
        x_text_R2_train, y_text_R2_train = xy_low + (0.05 + x_offset) * (xy_up - xy_low), \
                                           xy_low + 0.63 * (xy_up - xy_low)
        x_text_MAE_test, y_text_MAE_test = xy_low + (0.40 + x_offset) * (xy_up - xy_low), \
                                           xy_low + 0.17 * (xy_up - xy_low)
        x_text_R2_test, y_text_R2_test = xy_low + (0.40 + x_offset) * (xy_up - xy_low), \
                                         xy_low + 0.1 * (xy_up - xy_low)
    elif mode == "ppt":
        x_offset = 0.1
        x_text_MAE_train, y_text_MAE_train = xy_low + (0.02 + x_offset) * (xy_up - xy_low), \
                                             xy_low + 0.71 * (xy_up - xy_low)
        x_text_R2_train, y_text_R2_train = xy_low + (0.02 + x_offset) * (xy_up - xy_low), \
                                           xy_low + 0.63 * (xy_up - xy_low)
        x_text_MAE_test, y_text_MAE_test = xy_low + (0.3 + x_offset) * (xy_up - xy_low), \
                                           xy_low + 0.13 * (xy_up - xy_low)
        x_text_R2_test, y_text_R2_test = xy_low + (0.3 + x_offset) * (xy_up - xy_low), \
                                         xy_low + 0.05 * (xy_up - xy_low)

    dfs = []
    for i in range(5):
        plt.clf()

        res_file = res_path + "model_pred_" + str(i) + ".csv"
        df = pd.read_csv(res_file, usecols=["Target", "pred_Target", "set"])

        if not if_merge:
            df_train, df_test = df[df["set"] == "train"], df[df["set"] == "test"]
            y_scale = 1000 if "RebDuty" in perf else 1
            x_train, y_train = df_train["Target"] / y_scale, df_train["pred_Target"] / y_scale
            x_test, y_test = df_test["Target"] / y_scale, df_test["pred_Target"] / y_scale

            plt.scatter(x_train, y_train, c="red", s=dot_size, alpha=0.7, linewidths=0)
            plt.scatter(x_test, y_test, c="blue", s=dot_size, alpha=0.7, linewidths=0)
            plt.legend(labels=["Training set", "Test set"], prop={"size": legend_size})

            plt.xlabel(xlabel_dict[perf], size=label_size)
            plt.ylabel(ylabel_dict[perf], size=label_size)
            plt.xlim(xy_low, xy_up)
            plt.ylim(xy_low, xy_up)
            if xticks_dict[perf] is not None:
                plt.xticks(xticks_dict[perf])
                plt.yticks(xticks_dict[perf])
            plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)

            x_offset = 0.1 if "DIST" in perf else 0
            s1 = "MAE = " + str("{:.4f}".format(MAE_train[i] / y_scale)) + unit_dict[perf]
            plt.text(x_text_MAE_train, y_text_MAE_train, s1, c="red", fontsize=text_size)
            s2 = "R$^2$ = " + str("{:.4f}".format(R2_train[i]))
            plt.text(x_text_R2_train, y_text_R2_train, s2, c="red", fontsize=text_size)
            s3 = "MAE = " + str("{:.4f}".format(MAE_test[i] / y_scale)) + unit_dict[perf]
            plt.text(x_text_MAE_test, y_text_MAE_test, s3, c="blue", fontsize=text_size)
            s4 = "R$^2$ = " + str("{:.4f}".format(R2_test[i]))
            plt.text(x_text_R2_test, y_text_R2_test, s4, c="blue", fontsize=text_size)
            # x_offset = 0.1 if "DIST" in perf else 0
            # s1 = "MAE = " + str("{:.3f}".format(MAE_train[i] / y_scale)) + unit_dict[perf]
            # plt.text(xy_low + (0.05 + x_offset) * (xy_up - xy_low), xy_up - 0.3 * (xy_up - xy_low), s1, c="red",
            #          fontsize=text_size)
            # s2 = "R$^2$ = " + str("{:.4f}".format(R2_train[i]))
            # plt.text(xy_low + (0.05 + x_offset) * (xy_up - xy_low), xy_up - 0.37 * (xy_up - xy_low), s2, c="red",
            #          fontsize=text_size)
            # s3 = "MAE = " + str("{:.3f}".format(MAE_test[i] / y_scale)) + unit_dict[perf]
            # plt.text(xy_up - (0.55 - x_offset) * (xy_up - xy_low), xy_low + 0.17 * (xy_up - xy_low), s3, c="blue",
            #          fontsize=text_size)
            # s4 = "R$^2$ = " + str("{:.4f}".format(R2_test[i]))
            # plt.text(xy_up - (0.55 - x_offset) * (xy_up - xy_low), xy_low + 0.1 * (xy_up - xy_low), s4, c="blue",
            #          fontsize=text_size)
            plt.savefig(path + l + "_" + perf + "_" + str(i))
        else:
            df = df.rename(columns={"pred_Target": "pred_Target_" + str(i)})
            if i == 0:
                dfs.append(df)
            else:
                dfs.append(df.drop(columns=["Target", "set"]))

    if if_merge:
        DF = pd.concat(dfs, axis=1)
        print(f"# of data points: {len(DF)}")
        pred_columns = ["pred_Target_" + str(i) for i in range(5)]
        DF["pred_Target_mean"] = DF[pred_columns].mean(axis=1)
        DF["pred_Target_std"] = DF[pred_columns].std(axis=1)

        DF_train, DF_test = DF[DF["set"] == "train"], DF[DF["set"] == "test"]

        y_scale = 1000 if "RebDuty" in perf else 1
        x_train, y_train, er_train = DF_train["Target"] / y_scale, DF_train["pred_Target_mean"] / y_scale, \
                                     DF_train["pred_Target_std"] / y_scale
        x_test, y_test, er_test = DF_test["Target"] / y_scale, DF_test["pred_Target_mean"] / y_scale, \
                                  DF_test["pred_Target_std"] / y_scale
        plt.scatter(x_train, y_train, c="red", s=dot_size, alpha=0.7, linewidths=0, zorder=1)
        plt.errorbar(x_train, y_train, yerr=er_train, c="None", ecolor="red", elinewidth=0.8, zorder=2)
        plt.scatter(x_test, y_test, c="blue", s=dot_size, alpha=0.7, linewidths=0, zorder=3)
        plt.errorbar(x_test, y_test, yerr=er_test, c="None", ecolor="blue", elinewidth=0.8, zorder=4)
        plt.legend(labels=["Training set", "Test set"], prop={"size": legend_size})

        plt.xlabel(xlabel_dict[perf], size=label_size)
        plt.ylabel(ylabel_dict[perf], size=label_size)
        plt.xlim(xy_low, xy_up)
        plt.ylim(xy_low, xy_up)
        if xticks_dict[perf] is not None:
            plt.xticks(xticks_dict[perf])
            plt.yticks(xticks_dict[perf])
        plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)

        x_offset = 0.04 if "DIST" in perf else 0
        x_initial = 0.01 if "DIST" in perf else 0.03
        s1 = "MAE = " + str("{:.4f}".format(MAE_train_mean / y_scale)) + unit_dict[perf]
        plt.text(x_text_MAE_train, y_text_MAE_train, s1, c="red", fontsize=text_size)
        s2 = "R$^2$ = " + str("{:.4f}".format(R2_train_mean))
        plt.text(x_text_R2_train, y_text_R2_train, s2, c="red", fontsize=text_size)
        s3 = "MAE = " + str("{:.4f}".format(MAE_test_mean / y_scale)) + unit_dict[perf]
        plt.text(x_text_MAE_test, y_text_MAE_test, s3, c="blue", fontsize=text_size)
        s4 = "R$^2$ = " + str("{:.4f}".format(R2_test_mean))
        plt.text(x_text_R2_test, y_text_R2_test, s4, c="blue", fontsize=text_size)
        plt.savefig(path + l + "_" + perf)


" ---------------------------------------------------------- "


def parity_c(params, res_main_path, path, model_spec, if_merge=False):
    # plt.rcParams["figure.figsize"] = (4, 3)
    plt.clf()
    y_scale = 1000
    if params["task"] == "solvent_process":
        xlabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                       "DIST_C4H6_T2": "C$_4$H$_6$ purity",
                       "RebDuty_T1": "EDC heat duty (MW)",
                       "RebDuty_T2": "SRC heat duty (MW)"}
        ylabel_dict = {"DIST_C4H8_T1": "Predicted C$_4$H$_8$ purity",
                       "DIST_C4H6_T2": "Predicted C$_4$H$_6$ purity",
                       "RebDuty_T1": "Predicted EDC heat duty (MW)",
                       "RebDuty_T2": "Predicted SRC heat duty (MW)"}
        xylim_dict = {"DIST_C4H8_T1": extend_xylim(0.4, 1),  # extend_xylim(0.48, 1)
                      "DIST_C4H6_T2": extend_xylim(0.2, 1),
                      "RebDuty_T1": extend_xylim(0 / y_scale, 200000 / y_scale),
                      "RebDuty_T2": extend_xylim(0 / y_scale, 50000 / y_scale)}
        diagonal_dict = {"DIST_C4H8_T1": (0, 2), "RebDuty_T1": (-20000 / y_scale, 220000 / y_scale),
                         "DIST_C4H6_T2": (0, 2), "RebDuty_T2": (-5000 / y_scale, 55000 / y_scale)}
        unit_dict = {"DIST_C4H8_T1": "",
                     "DIST_C4H6_T2": "",
                     "RebDuty_T1": " MW",
                     "RebDuty_T2": " MW"}
        xticks_dict = {"DIST_C4H8_T1": np.arange(0.4, 1.1, 0.1),
                       "DIST_C4H6_T2": np.arange(0.2, 1.1, 0.2),
                       "RebDuty_T1": np.arange(0, 210, 50),
                       "RebDuty_T2": np.arange(0, 60, 10)}

    l, perf = model_spec
    res_path = res_main_path + "/" + perf + "/"
    score_file = res_path + "final_model_his.csv"
    df_score = pd.read_csv(score_file)
    MAE_train, R2_train = df_score["MAE_train"].values, df_score["R2_train"].values
    MAE_test, R2_test = df_score["MAE_test"].values, df_score["R2_test"].values

    MAE_train_mean, MAE_train_std = np.average(MAE_train), np.std(MAE_train)
    R2_train_mean, R2_train_std = np.average(R2_train), np.std(R2_train)
    MAE_test_mean, MAE_test_std = np.average(MAE_test), np.std(MAE_test)
    R2_test_mean, R2_test_std = np.average(R2_test), np.std(R2_test)

    xy_low, xy_up = xylim_dict[perf]
    x_low, x_up = diagonal_dict[perf]

    if mode == "doc":
        x_offset = 0.15
        x_text_MAE, y_text_MAE = xy_low + x_offset * (xy_up - xy_low), xy_low + 0.85 * (xy_up - xy_low)
        x_text_R2, y_text_R2 = xy_low + x_offset * (xy_up - xy_low), xy_low + 0.78 * (xy_up - xy_low)
    elif mode == "ppt":
        x_offset = 0.15
        x_text_MAE, y_text_MAE = xy_low + x_offset * (xy_up - xy_low), xy_low + 0.86 * (xy_up - xy_low)
        x_text_R2, y_text_R2 = xy_low + x_offset * (xy_up - xy_low), xy_low + 0.78 * (xy_up - xy_low)

    def density(x, y):
        # x_mean, x_std = np.mean(x), np.std(x)
        # y_mean, y_std = np.mean(y), np.std(y)
        # x_nor = (x - x_mean) / x_std
        # y_nor = (y - y_mean) / y_std
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        return x, y, z

    dfs = []
    for i in range(5):
        plt.clf()

        res_file = res_path + "model_pred_" + str(i) + ".csv"
        df = pd.read_csv(res_file, usecols=["Target", "pred_Target", "set"])
        df = df.sample(5000, random_state=42)

        if not if_merge:
            df_train, df_test = df[df["set"] == "train"], df[df["set"] == "test"]
            y_scale = 1000 if "RebDuty" in perf else 1
            x_train, y_train = df_train["Target"].values / y_scale, df_train["pred_Target"].values / y_scale
            x_test, y_test = df_test["Target"].values / y_scale, df_test["pred_Target"].values / y_scale

            x_train, y_train, z_train = density(x_train, y_train)
            x_test, y_test, z_test = density(x_test, y_test)

            plt.scatter(x_train, y_train, c=z_train, cmap="jet", s=dot_size)
            plt.xlabel(xlabel_dict[perf], size=label_size)
            plt.ylabel(ylabel_dict[perf], size=label_size)
            plt.xlim(xy_low, xy_up)
            plt.ylim(xy_low, xy_up)
            if xticks_dict[perf] is not None:
                plt.xticks(xticks_dict[perf])
                plt.yticks(xticks_dict[perf])
            plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
            s1 = "MAE = " + str("{:.4f}".format(MAE_train[i] / y_scale)) + unit_dict[perf]
            plt.text(x_text_MAE, y_text_MAE, s1, c="red", fontsize=text_size)
            s2 = "R$^2$ = " + str("{:.4f}".format(R2_train[i]))
            plt.text(x_text_R2, y_text_R2, s2, c="red", fontsize=text_size)
            # cbar = plt.colorbar(pad=0.02)
            # cbar.ax.tick_params(labelsize=8)
            plt.savefig(path + l + "_" + perf + "_train_" + str(i))

            plt.clf()
            plt.scatter(x_test, y_test, c=z_test, cmap="jet", s=dot_size)
            plt.xlabel(xlabel_dict[perf], size=label_size)
            plt.ylabel(ylabel_dict[perf], size=label_size)
            plt.xlim(xy_low, xy_up)
            plt.ylim(xy_low, xy_up)
            if xticks_dict[perf] is not None:
                plt.xticks(xticks_dict[perf])
                plt.yticks(xticks_dict[perf])
            plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
            s3 = "MAE = " + str("{:.4f}".format(MAE_test[i] / y_scale)) + unit_dict[perf]
            plt.text(x_text_MAE, y_text_MAE, s3, c="blue", fontsize=text_size)
            s4 = "R$^2$ = " + str("{:.4f}".format(R2_test[i]))
            plt.text(x_text_R2, y_text_R2, s4, c="blue", fontsize=text_size)
            # cbar = plt.colorbar(pad=0.02)
            # cbar.ax.tick_params(labelsize=8)
            plt.savefig(path + l + "_" + perf + "_test_" + str(i))

            # plt.scatter(x_train, y_train, c="red", s=dot_size, alpha=0.7, linewidths=0)
            # plt.scatter(x_test, y_test, c="blue", s=dot_size, alpha=0.7, linewidths=0)
            # plt.legend(labels=["Training set", "Test set"], prop={"size": legend_size})

            # x_offset = 0.1 if "DIST" in perf else 0
            # s1 = "MAE = " + str("{:.3f}".format(MAE_train[i] / y_scale)) + unit_dict[perf]
            # plt.text(xy_low + (0.05 + x_offset) * (xy_up - xy_low), xy_up - 0.3 * (xy_up - xy_low), s1, c="red",
            #          fontsize=8.5)
            # s2 = "R$^2$ = " + str("{:.4f}".format(R2_train[i]))
            # plt.text(xy_low + (0.05 + x_offset) * (xy_up - xy_low), xy_up - 0.37 * (xy_up - xy_low), s2, c="red",
            #          fontsize=8.5)
            # s3 = "MAE = " + str("{:.3f}".format(MAE_test[i] / y_scale)) + unit_dict[perf]
            # plt.text(xy_up - (0.55 - x_offset) * (xy_up - xy_low), xy_low + 0.17 * (xy_up - xy_low), s3, c="blue",
            #          fontsize=8.5)
            # s4 = "R$^2$ = " + str("{:.4f}".format(R2_test[i]))
            # plt.text(xy_up - (0.55 - x_offset) * (xy_up - xy_low), xy_low + 0.1 * (xy_up - xy_low), s4, c="blue",
            #          fontsize=8.5)
        else:
            df = df.rename(columns={"pred_Target": "pred_Target_" + str(i)})
            if i == 0:
                dfs.append(df)
            else:
                dfs.append(df.drop(columns=["Target", "set"]))

    if if_merge:
        DF = pd.concat(dfs, axis=1)
        print(f"# of data points: {len(DF)}", flush=True)
        pred_columns = ["pred_Target_" + str(i) for i in range(5)]
        DF["pred_Target_mean"] = DF[pred_columns].mean(axis=1)
        DF["pred_Target_std"] = DF[pred_columns].std(axis=1)

        DF_train, DF_test = DF[DF["set"] == "train"], DF[DF["set"] == "test"]

        y_scale = 1000 if "RebDuty" in perf else 1
        x_train, y_train, er_train = DF_train["Target"].values / y_scale, DF_train["pred_Target_mean"].values / y_scale, \
                                     DF_train["pred_Target_std"] / y_scale
        x_test, y_test, er_test = DF_test["Target"].values / y_scale, DF_test["pred_Target_mean"].values / y_scale, \
                                  DF_test["pred_Target_std"] / y_scale

        x_train, y_train, z_train = density(x_train, y_train)
        x_test, y_test, z_test = density(x_test, y_test)

        plt.scatter(x_train, y_train, c=z_train, cmap="jet", s=dot_size)
        plt.xlabel(xlabel_dict[perf], size=label_size)
        plt.ylabel(ylabel_dict[perf], size=label_size)
        plt.xlim(xy_low, xy_up)
        plt.ylim(xy_low, xy_up)
        if xticks_dict[perf] is not None:
            plt.xticks(xticks_dict[perf])
            plt.yticks(xticks_dict[perf])
        plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
        s1 = "MAE = " + str("{:.4f}".format(MAE_train_mean / y_scale)) + unit_dict[perf]
        plt.text(x_text_MAE, y_text_MAE, s1, c="red", fontsize=text_size)
        s2 = "R$^2$ = " + str("{:.4f}".format(R2_train_mean))
        plt.text(x_text_R2, y_text_R2, s2, c="red", fontsize=text_size)
        # cbar = plt.colorbar(pad=0.02)
        # cbar.ax.tick_params(labelsize=8)
        plt.savefig(path + l + "_" + perf + "_train")

        plt.clf()
        plt.scatter(x_test, y_test, c=z_test, cmap="jet", s=dot_size)
        plt.xlabel(xlabel_dict[perf], size=label_size)
        plt.ylabel(ylabel_dict[perf], size=label_size)
        plt.xlim(xy_low, xy_up)
        plt.ylim(xy_low, xy_up)
        if xticks_dict[perf] is not None:
            plt.xticks(xticks_dict[perf])
            plt.yticks(xticks_dict[perf])
        plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
        s3 = "MAE = " + str("{:.4f}".format(MAE_test_mean / y_scale)) + unit_dict[perf]
        plt.text(x_text_MAE, y_text_MAE, s3, c="blue", fontsize=text_size)
        s4 = "R$^2$ = " + str("{:.4f}".format(R2_test_mean))
        plt.text(x_text_R2, y_text_R2, s4, c="blue", fontsize=text_size)
        # cbar = plt.colorbar(pad=0.02)
        # cbar.ax.tick_params(labelsize=8)
        plt.savefig(path + l + "_" + perf + "_test")

        # plt.scatter(x_train, y_train, c="red", s=dot_size, alpha=0.7, linewidths=0, zorder=1)
        # plt.errorbar(x_train, y_train, yerr=er_train, c="None", ecolor="red", elinewidth=0.8, zorder=2)
        # plt.scatter(x_test, y_test, c="blue", s=dot_size, alpha=0.7, linewidths=0, zorder=3)
        # plt.errorbar(x_test, y_test, yerr=er_test, c="None", ecolor="blue", elinewidth=0.8, zorder=4)
        # plt.legend(labels=["Training set", "Test set"], prop={"size": legend_size})

        # plt.xlabel(xlabel_dict[perf], size=label_size)
        # plt.ylabel(ylabel_dict[perf], size=label_size)
        # xy_low, xy_up = xylim_dict[perf]
        # plt.xlim(xy_low, xy_up)
        # plt.ylim(xy_low, xy_up)
        # x_low, x_up = diagonal_dict[perf]
        # plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)

        # x_offset = 0.04 if "DIST" in perf else 0
        # x_initial = 0.01 if "DIST" in perf else 0.03
        # s1 = "MAE = " + str("{:.3f}".format(MAE_train_mean / y_scale)) + " ± " + "{:.3f}".format(
        #     MAE_train_std / y_scale) + unit_dict[perf]
        # plt.text(xy_low + (x_initial + x_offset) * (xy_up - xy_low), xy_up - 0.28 * (xy_up - xy_low), s1, c="red",
        #          fontsize=8)
        # s2 = "R$^2$ = " + str("{:.4f}".format(R2_train_mean)) + " ± " + str("{:.4f}".format(R2_train_std))
        # plt.text(xy_low + (x_initial + x_offset) * (xy_up - xy_low), xy_up - 0.35 * (xy_up - xy_low), s2, c="red",
        #          fontsize=8)
        # s3 = "MAE = " + str("{:.3f}".format(MAE_test_mean / y_scale)) + " ± " + "{:.3f}".format(
        #     MAE_test_std / y_scale) + unit_dict[perf]
        # plt.text(xy_up - (0.65 - x_offset) * (xy_up - xy_low), xy_low + 0.17 * (xy_up - xy_low), s3, c="blue",
        #          fontsize=8)
        # s4 = "R$^2$ = " + str("{:.4f}".format(R2_test_mean)) + " ± " + str("{:.4f}".format(R2_test_std))
        # plt.text(xy_up - (0.65 - x_offset) * (xy_up - xy_low), xy_low + 0.1 * (xy_up - xy_low), s4, c="blue",
        #          fontsize=8)
        # plt.savefig(path + l + "_" + perf)


" ---------------------------------------------------------- "


def Figure_1a(res_main_path, merge_state=False):
    """ # Generate the parity plots for ML models """
    print("Figure 1a:", merge_state, flush=True)
    for l in ["nonlinear"]:
        for perf in ["DIST_C4H8_T1", "RebDuty_T1"]:
            params = get_param("solvent", (perf, "regression", "T1"))
            model_spec = (l, perf)
            sub_viz_path = create_directory(params["viz_path"] + mode + "/1_parity/")
            parity_a(params, res_main_path, sub_viz_path, model_spec,
                     merge_state)  # True: five plots for five models; False: one plot
    plt.close()


def Figure_1b(res_main_path, ci, merge_state=False):
    """ # Generate the parity plots for ML models """
    print("Figure 1b:", ci, merge_state, flush=True)
    params = get_param("process")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/1_parity/")
    for l in ["nonlinear"]:
        OFs = ["DIST_C4H8_T1", "RebDuty_T1"] if ci == "T1" else ["DIST_C4H6_T2", "RebDuty_T2"]
        for perf in OFs:
            model_spec = (l, perf)
            parity_a(params, res_main_path, sub_viz_path, model_spec,
                     merge_state)  # True: five plots for five models; False: one plot
    plt.close()


def Figure_1c(res_main_path, ci, merge_state=False):
    """ # Generate the parity plots for ML models """
    print("Figure 1c:", ci, merge_state, flush=True)
    params = get_param("solvent_process")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/1_parity/")
    for l in ["nonlinear"]:
        OFs = ["DIST_C4H8_T1", "RebDuty_T1"] if ci == "T1" else ["DIST_C4H6_T2", "RebDuty_T2"]
        for perf in OFs:
            model_spec = (l, perf)
            parity_c(params, res_main_path, sub_viz_path, model_spec,
                     merge_state)  # True: five plots for five models; False: one plot
    plt.close()


" #################################################################################################################### "


def pareto_vs_iteration_a(params, res_main_path, path, model_spec, if_overlap=False):
    plt.clf()
    l, c, ci = model_spec

    str_constr = "" if c is None else "_constr"
    res_path = res_main_path + "/optimization_" + ci + str_constr + "/optimization.pickle"

    print(res_path, flush=True)
    res_history = pickle_load(res_path)

    if params["task"] == "solvent":
        iters_dict = {"nonlinear": [1, 3, 5, 6, 9, 10, 90]}
        xylim_dict = {"nonlinear": (extend_xylim(0.0, 0.04), extend_xylim(-3, 3))}
        mean, std = 24037.318223076927, 7712.179331204997
        xticks = None
    elif params["task"] == "process":
        if ci == "T1T2":
            iters_dict = {"nonlinear": [1, 4, 6, 10, 25, 50, 100]}
            xylim_dict = {"nonlinear": ((0.6, 1), extend_xylim(-1.6, -0.6))}
            mean, std = 33480.917166413456 + 16338.2463890172, np.sqrt(
                16933.223558474107 ** 2 + 3612.867089587996 ** 2)
            xticks = None  # np.arange(0, 0.01, 0.003)
    elif params["task"] == "solvent_process":
        if ci == "T1T2":
            iters_dict = {"nonlinear": [1, 6, 7, 10, 20, 30, 40, 50, 75]}  # [1, 2, 4, 5, 7, 9, 110]
            xylim_dict = {"nonlinear": ((0.5, 1), extend_xylim(-1.1, -0.5))}
            mean, std = 41614.91843103464 + 11911.925773664572, np.sqrt(
                26906.900856946824 ** 2 + 5994.026928442846 ** 2)
            xticks = np.arange(0.5, 1.05, 0.1)

    " # Visualize the Pareto front at several iterations "
    fig = plt.figure()
    for i, e in enumerate(res_history):
        if not if_overlap:
            plt.clf()
        points = np.array([ei.F for ei in e.opt])
        if if_overlap:
            iters = np.array(iters_dict[l] + [len(res_history)])
            if i in iters - 1:
                plt.scatter(points[:, 0], points[:, 1], s=dot_size)
        else:
            plt.scatter(points[:, 0], points[:, 1], s=dot_size)
        plt.xlabel("Objective function 1", size=label_size)
        plt.ylabel("Objective function 2", size=label_size)
        x_lim, y_lim = xylim_dict[l]
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        if xticks is not None: plt.xticks(xticks)
        if not if_overlap:
            label = " ".join(["Iteration", str(i + 1)])
            plt.legend(labels=[label], prop={"size": legend_size})
            plt.savefig(path + "iteration_" + str(i + 1))
    if if_overlap:
        dec_value = 0 if params["task"] == "solvent_process" else 0
        # if mode == "doc":
        #     plt.legend(labels=[" ".join(["Iteration", str(i)]) for i in iters], prop={"size": legend_size - dec_value})
        # elif mode == "ppt":
        plt.legend(labels=[" ".join(["Iteration", str(i)]) for i in iters], prop={"size": legend_size - dec_value},
                   loc="upper left", bbox_to_anchor=(1, 1))
        plt.axhline(y=(0 - mean) / std, c="k", linestyle="--", zorder=0)
        plt.axvline(x=0, c="k", linestyle="--", zorder=0)
        plt.axvline(x=1, c="k", linestyle="--", zorder=0)
        plt.plot(0, (0 - mean) / std, "*", c="red", ms=10)
        plt.savefig(path + l + "_total_" + ci + str_constr)

    " # Animation of the Pareto front at every iteration "
    plt.clf()
    fig = plt.figure()
    scat, *_ = plt.plot([], [], "o")

    def update(i):
        points = np.array([ei.F for ei in res_history[i].opt])
        scat.set_data(points[:, 0], points[:, 1])
        label = " ".join(["Iteration", str(i + 1)])
        plt.legend(labels=[label], prop={"size": legend_size}, loc=1)

    plt.xlabel("Objective function 1", size=label_size)
    plt.ylabel("Objective function 2", size=label_size)
    x_lim, y_lim = xylim_dict[l]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    ani = FuncAnimation(fig, update, np.arange(len(res_history)))
    writer = PillowWriter(fps=10)
    fig.tight_layout()
    ani.save(path + l + "_animation_" + ci + str_constr + ".gif", writer=writer, savefig_kwargs={"transparent": False})


" ---------------------------------------------------------- "


def pareto_vs_iteration_d(params, res_main_path, path, model_spec, if_overlap=False):
    plt.clf()
    l, c, ci, solvent = model_spec

    str_constr = "" if c is None else "_constr"
    res_path = res_main_path + "/optimization_" + ci + str_constr + f"/optimization_{solvent}.pickle"

    print(res_path, flush=True)
    res_history = pickle_load(res_path)

    if params["task"] == "solvent_process":
        if ci == "T1T2":
            iters_dict = {"nonlinear": []}  # [1, 2, 4, 5, 7, 9, 110]
            x_lim = extend_xylim(0.4, 1)
            y_lim = extend_xylim(-1.1, -0.5)
            mean, std = 41614.91843103464 + 11911.925773664572, np.sqrt(
                26906.900856946824 ** 2 + 5994.026928442846 ** 2)
            # xticks = np.arange(0, 0.13, 0.02)

    " # Visualize the Pareto front at several iterations "
    # fig = plt.figure()
    for i, e in enumerate(res_history):
        if not if_overlap:
            plt.clf()
        points = np.array([ei.F for ei in e.opt])
        if if_overlap:
            iters = np.array(iters_dict[l] + [len(res_history)])
            if i in iters - 1:
                plt.scatter(points[:, 0], points[:, 1], s=dot_size)
        else:
            plt.scatter(points[:, 0], points[:, 1], s=dot_size)
        plt.xlabel("Objective function 1", size=label_size)
        plt.ylabel("Objective function 2", size=label_size)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        # if xticks is not None: plt.xticks(xticks)
        if not if_overlap:
            label = " ".join(["Iteration", str(i + 1)])
            plt.legend(labels=[label], prop={"size": legend_size})
            plt.savefig(path + "iteration_" + str(i + 1))
    if if_overlap:
        dec_value = 1 if params["task"] == "solvent_process" else 0
        # if mode == "doc":
        #     plt.legend(labels=[" ".join(["Iteration", str(i)]) for i in iters], prop={"size": legend_size - dec_value})
        # elif mode == "ppt":
        # plt.legend(labels=[" ".join(["Iteration", str(i)]) for i in iters], prop={"size": legend_size - dec_value},
        #            loc="upper left", bbox_to_anchor=(1, 1))
        plt.axhline(y=(0 - mean) / std, c="k", linestyle="--", zorder=0)
        plt.axvline(x=0, c="k", linestyle="--", zorder=0)
        # plt.axvline(x=1, c="k", linestyle="--", zorder=0)
        plt.plot(0, (0 - mean) / std, "*", c="red", ms=10)
        plt.savefig(path + l + "_total_" + ci)

    " # Animation of the Pareto front at every iteration "
    plt.clf()
    fig = plt.figure()
    scat, *_ = plt.plot([], [], "o")

    def update(i):
        points = np.array([ei.F for ei in res_history[i].opt])
        scat.set_data(points[:, 0], points[:, 1])
        label = " ".join(["Iteration", str(i + 1)])
        plt.legend(labels=[label], prop={"size": legend_size}, loc=1)

    plt.xlabel("Objective function 1", size=label_size)
    plt.ylabel("Objective function 2", size=label_size)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    ani = FuncAnimation(fig, update, np.arange(len(res_history)))
    writer = PillowWriter(fps=10)
    fig.tight_layout()
    ani.save(path + l + "_animation_" + ci + ".gif", writer=writer, savefig_kwargs={"transparent": False})


" ---------------------------------------------------------- "


def Figure_2a(res_main_path):
    " # Visualize the Pareto front in multi-objective optimization "
    print("Figure 2a")
    params = get_param("solvent")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/2_pareto/")
        pareto_vs_iteration_a(params, res_main_path, sub_viz_path, (l, c, "T1"), True)
    plt.close()


def Figure_2b(res_main_path, ci):
    " # Visualize the Pareto front in multi-objective optimization "
    print("Figure 2b:", ci)
    params = get_param("process")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/2_pareto/")
        pareto_vs_iteration_a(params, res_main_path, sub_viz_path, (l, c, ci), True)
    plt.close()


def Figure_2c(res_main_path, ci):
    " # Visualize the Pareto front in multi-objective optimization "
    print("Figure 2c:", ci)
    params = get_param("solvent_process")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/2_pareto/")
        pareto_vs_iteration_a(params, res_main_path, sub_viz_path, (l, c, ci), True)
    plt.close()


def Figure_2d(res_main_path, ci):
    " # Visualize the Pareto front in multi-objective optimization "
    print("Figure 2d:", ci)
    params = get_param("solvent_process")
    solvents = ["0_C5H8O2-D1", "1_C4H7NO-E2", "3_C4H6O2-N1", "5_C5H8O3-D2", "6_C4H6O3", "7_C8H8O-D3", "8_C4H10N2"]
    for l, c in itertools.product(["nonlinear"], [None]):
        for solvent in solvents:
            sub_viz_path = create_directory(params["viz_path"] + mode + f"/2_pareto_{solvent}/")
            pareto_vs_iteration_d(params, res_main_path, sub_viz_path, (l, c, ci, solvent), True)
            plt.close()


" #################################################################################################################### "


def cand_prop_viz_pareto_a(params, res_main_path, path, model_spec):
    plt.clf()
    y_scale = 1000
    l, c, ci = model_spec
    if params["task"] == "solvent":
        xlim_dict = {"nonlinear": extend_xylim(0.9, 1)}
        ylim_dict = {"nonlinear": extend_xylim(0000 / y_scale, 50000 / y_scale)}
        x, y = 0.995466491, 16660.6957 / y_scale
        x_pred, y_pred = 0.972904214, 18375.1172 / y_scale
        y_mean, y_std = 24037.318223076927, 7712.179331204997
        xticks = np.arange(0.9, 1.01, 0.02)
    elif params["task"] == "process":
        if ci == "T1":
            xlim_dict = {"nonlinear": extend_xylim(0.9, 1)}
            ylim_dict = {"nonlinear": extend_xylim(1800 / y_scale, 3200 / y_scale)}
            x, y = 0.992693432, 2596.52976 / y_scale
            x_pred, y_pred = 0.985857306, 2599.23028 / y_scale
            # y_mean, y_std = 5019.632200595847, 3023.966962321585
            # y_mean, y_std = 3883.816608040528, 1762.6347594241297
        elif ci == "T2":
            xlim_dict = {"nonlinear": extend_xylim(0.995, 1)}
            ylim_dict = {"nonlinear": extend_xylim(2400 / y_scale, 3000 / y_scale)}
            # y_mean, y_std = 3867.0632216546755, 1260.007701493469
            # y_mean, y_std = 3090.2694132801953, 1044.3813734345836
        elif ci == "T1T2":
            xlim_dict = {"nonlinear": extend_xylim(0.992, 0.998)}
            ylim_dict = {"nonlinear": extend_xylim(20000 / y_scale, 40000 / y_scale)}
            y_mean, y_std = 3883.816608040528 + 3090.2694132801953, np.sqrt(
                1762.6347594241297 ** 2 + 1044.3813734345836 ** 2)
        xticks = np.arange(0.992, 0.999, 0.002)
        yticks = np.arange(20, 45, 5)

    dot_color = "blue" if params["task"] == "solvent" else "tab:grey"

    str_constr = "" if c is None else "_constr"
    res_path = res_main_path + "/optimization_" + ci + str_constr + "/"

    if params["task"] == "solvent":
        df_cand = pd.read_csv(res_path + "identified_cand.csv")
        df_cand = df_cand[df_cand["DIST_C4H8"] >= 0.5]
        x_cand, y_cand = df_cand["f_ave_C4H8"].values, df_cand["f_ave_Duty"].values / y_scale
        x_true, y_true = df_cand["DIST_C4H8"].values, df_cand["RebDuty"].values / y_scale

    elif params["task"] == "process":
        df_cand = pd.read_csv(res_path + "process_evaluation" + str_constr + ".csv")

        if ci == "T1":
            df_cand = df_cand[df_cand["hasERROR_" + ci] == False]
            x_cand, y_cand = df_cand["f_ave_C4H8"].values, df_cand["f_ave_Duty"].values / y_scale
            x_true, y_true = df_cand["DIST_C4H8_" + ci].values, df_cand["RebDuty_" + ci].values / y_scale
        elif ci == "T2":
            df_cand = df_cand[df_cand["hasERROR_" + ci] == False]
            x_cand, y_cand = df_cand["f_ave_C4H6"].values, df_cand["f_ave_Duty"].values / y_scale
            x_true, y_true = df_cand["DIST_C4H6_" + ci].values, df_cand["RebDuty_" + ci].values / y_scale
        elif ci == "T1T2":
            df_cand = df_cand[df_cand["hasERROR"] == False]
            x_cand, y_cand = df_cand["f_ave_DIST_C4H8_T1"].values, \
                             (df_cand["f_ave_RebDuty_T1"].values + df_cand["f_ave_RebDuty_T2"].values) / y_scale
            x_true, y_true = df_cand["DIST_C4H8_T1"].values, df_cand["RebDuty_total"].values / y_scale
    #
    # elif params["task"] == "solvent_process":
    #     df_cand = pd.read_csv(res_path + "process_evaluation_" + ci + str_constr + ".csv")
    #     if ci == "T1T2":
    #         x_cand, y_cand = df_cand["f_ave_DIST_C4H8_T1"].values, \
    #                          (df_cand["f_ave_RebDuty_T1"].values + df_cand["f_ave_RebDuty_T2"].values) / y_scale
    #         x_true, y_true = df_cand["DIST_C4H8_T1"].values, df_cand["RebDuty_total"].values / y_scale
    #     xticks = None

    print(f"# of data points: {len(x_cand)}", flush=True)
    plt.scatter(x_cand, y_cand, c=dot_color, s=dot_size, zorder=1, label="Model-based estimation")
    # plt.axhline(y=0, c="k", linestyle="--", zorder=0)
    # plt.axvline(x=1, c="k", linestyle="--", zorder=0)
    plt.plot(1, 0, "*", c="red", ms=10)

    F = np.load(res_path + "F_best.npy")
    F = np.array([[1 - f[0], f[1] * y_std + y_mean] for f in F])
    x_pareto, y_pareto = F[:, 0], F[:, 1] / y_scale
    if params["task"] == "solvent": plt.scatter(x_pareto, y_pareto, c="tab:grey", s=dot_size, zorder=0)

    if "T2" not in ci: plt.scatter(x_pred, y_pred, c="limegreen", s=dot_size, zorder=5, label="Benchmark"), \
                       print(x_pred, y_pred, flush=True)
    plt.xlabel("Predicted purity", size=label_size)
    plt.ylabel("Predicted heat duty (MW)", size=label_size)
    plt.xlim(xlim_dict[l])
    plt.ylim(ylim_dict[l])
    # plt.legend(prop={"size": legend_size})
    if params["task"] == "solvent":
        if xticks is not None: plt.xticks(xticks)
        plt.savefig(path + l + "_" + ci + "_pred")
    else:
        plt.scatter(x_true, y_true, c="blue", s=dot_size, label="Process simulation")
        # plt.legend(prop={"size: legend_size})
        if "T2" not in ci: plt.scatter(x, y, c="limegreen", s=dot_size, zorder=5, label="Benchmark"), \
                           print(x, y, flush=True)

        plt.xlabel("C$_4$H$_8$ purity", size=label_size)
        plt.ylabel("Total heat duty (MW)", size=label_size)
        plt.xlim(xlim_dict[l])
        plt.ylim(ylim_dict[l])
        if xticks is not None: plt.xticks(xticks)
        if yticks is not None: plt.yticks(yticks)

        plt.legend(prop={"size": legend_size})
        plt.savefig(path + l + "_" + ci + str_constr + "_comp")

    plt.clf()
    plt.scatter(x_true, x_cand, c="blue", s=dot_size)
    if "T2" not in ci: plt.scatter(x, x_pred, c="limegreen", s=dot_size, zorder=1)
    plt.xlabel("C$_4$H$_8$ purity", size=label_size)
    plt.ylabel("Predicted purity", size=label_size)
    plt.xlim(xlim_dict[l])
    plt.ylim(xlim_dict[l])
    if xticks is not None: plt.xticks(xticks), plt.yticks(xticks)
    plt.plot([0, 100], [0, 100], color="k", zorder=0)
    # plt.scatter(y_true, y_cand)
    plt.savefig(path + "predicted_purity_" + ci + str_constr)

    plt.clf()
    plt.scatter(y_true, y_cand, c="blue", s=dot_size)
    if "T2" not in ci: plt.scatter(y, y_pred, c="limegreen", s=dot_size, zorder=1)
    plt.xlabel("Reboiler heat duty (MW)", size=label_size)
    plt.ylabel("Predicted heat duty (MW)", size=label_size)
    plt.xlim(ylim_dict[l])
    plt.ylim(ylim_dict[l])
    if xticks is not None: plt.xticks(yticks), plt.yticks(yticks)
    plt.plot([-10, 100], [-10, 100], color="k", zorder=0)
    # plt.scatter(y_true, y_cand)

    plt.savefig(path + "predicted_duty_" + ci + str_constr)


" ---------------------------------------------------------- "


def cand_prop_viz_pareto_c(params, res_main_path, path, model_spec):
    plt.clf()
    y_scale = 1000
    l, c, ci = model_spec

    if params["task"] == "solvent_process":
        if ci == "T1T2":
            xlim_dict = {"nonlinear": extend_xylim(0.6, 1)}
            ylim_dict = {"nonlinear": extend_xylim(10000 / y_scale, 50000 / y_scale)}
            y_mean, y_std = 3883.816608040528 + 3090.2694132801953, np.sqrt(
                1762.6347594241297 ** 2 + 1044.3813734345836 ** 2)
    dot_color = "blue" if params["task"] == "solvent" else "tab:grey"

    str_constr = "" if c is None else "_constr"
    res_path = res_main_path + "/optimization_" + ci + str_constr + "/"

    if params["task"] == "solvent_process":
        df_cand = pd.read_csv(res_path + f"process_evaluation.csv")
        if ci == "T1T2":
            df_cand = df_cand[df_cand["hasERROR"] == False]
            df_cand = df_cand[df_cand["DIST_C4H8_T1"] >= 0.5]
            x_cand, y_cand = df_cand["f_ave_DIST_C4H8_T1"].values, \
                             (df_cand["f_ave_RebDuty_T1"].values + df_cand["f_ave_RebDuty_T2"].values) / y_scale
            x_true, y_true = df_cand["DIST_C4H8_T1"].values, df_cand["RebDuty_total"].values / y_scale
        xticks = None

    print(f"# of data points: {len(x_cand)}", flush=True)
    plt.scatter(x_cand, y_cand, c=dot_color, s=dot_size, zorder=1, label="Model-based estimation")
    # plt.axhline(y=0, c="k", linestyle="--", zorder=0)
    # plt.axvline(x=1, c="k", linestyle="--", zorder=0)
    # plt.plot(1, 0, "*", c="red", ms=10)

    F = np.load(res_path + "F_best.npy")
    F = np.array([[1 - f[0], f[1] * y_std + y_mean] for f in F])
    x_pareto, y_pareto = F[:, 0], F[:, 1] / y_scale
    if params["task"] == "solvent": plt.scatter(x_pareto, y_pareto, c="tab:grey", s=dot_size, zorder=0)

    # if "T2" not in ci: plt.scatter(x_pred, y_pred, c="limegreen", s=dot_size, zorder=5, label="Benchmark"), \
    #                    print(x_pred, y_pred, flush=True)
    plt.xlabel("Predicted purity", size=label_size)
    plt.ylabel("Predicted heat duty (MW)", size=label_size)
    plt.xlim(xlim_dict[l])
    plt.ylim(ylim_dict[l])
    # plt.legend(prop={"size": legend_size})
    if params["task"] == "solvent":
        if xticks is not None: plt.xticks(xticks)
        plt.savefig(path + l + "_" + ci + "_pred")
    else:
        plt.scatter(x_true, y_true, c="blue", s=dot_size, label="Process simulation")
        # plt.legend(prop={"size: legend_size})
        # if "T2" not in ci: plt.scatter(x, y, c="limegreen", s=dot_size, zorder=5, label="Benchmark"), \
        #                    print(x, y, flush=True)

        plt.xlabel("C$_4$H$_8$ purity", size=label_size)
        plt.ylabel("Reboiler heat duty (MW)", size=label_size)
        plt.xlim(xlim_dict[l])
        plt.ylim(ylim_dict[l])
        plt.legend(loc="upper left", prop={"size": legend_size})
        plt.savefig(path + l + "_" + ci + str_constr + "_comp")

    x_low, x_up = xlim_dict[l]
    y_low, y_up = ylim_dict[l]
    if mode == "doc":
        x_text_1, y_text_1 = x_low + 0.1 * (x_up - x_low), x_low + 0.85 * (x_up - x_low)
        x_text_2, y_text_2 = y_low + 0.1 * (y_up - y_low), y_low + 0.85 * (y_up - y_low)
    elif mode == "ppt":
        x_text_1, y_text_1 = x_low + 0.1 * (x_up - x_low), x_low + 0.85 * (x_up - x_low)
        x_text_2, y_text_2 = y_low + 0.1 * (y_up - y_low), y_low + 0.85 * (y_up - y_low)

    plt.clf()
    plt.scatter(x_true, x_cand, c="blue", s=dot_size)
    # if "T2" not in ci: plt.scatter(x, x_pred, c="limegreen", s=dot_size, zorder=1)
    plt.xlabel("C$_4$H$_8$ purity", size=label_size)
    plt.ylabel("Predicted purity", size=label_size)
    plt.xlim(xlim_dict[l])
    plt.ylim(xlim_dict[l])
    if xticks is not None: plt.xticks(xticks)
    plt.plot([-1, 2], [-1, 2], color="k", zorder=0)
    s1 = "MAE = " + str("{:.4f}".format(mean_absolute_error(x_true, x_cand)))
    plt.text(x_text_1, y_text_1, s1, c="k", fontsize=text_size)
    # plt.scatter(y_true, y_cand)
    plt.savefig(path + "predicted_purity_" + ci + str_constr)

    plt.clf()
    plt.scatter(y_true, y_cand, c="blue", s=dot_size)
    # if "T2" not in ci: plt.scatter(y, y_pred, c="limegreen", s=dot_size, zorder=1)
    plt.xlabel("Reboiler heat duty (MW)", size=label_size)
    plt.ylabel("Predicted heat duty (MW)", size=label_size)
    plt.xlim(ylim_dict[l])
    plt.ylim(ylim_dict[l])
    # plt.xticks(np.arange(1, 9))
    if xticks is not None: plt.xticks(xticks)
    plt.plot([-10, 100], [-10, 100], color="k", zorder=0)
    s2 = "MAE = " + str("{:.4f}".format(mean_absolute_error(y_true, y_cand))) + " MW"
    plt.text(x_text_2, y_text_2, s2, c="k", fontsize=text_size)

    # plt.scatter(y_true, y_cand)

    plt.savefig(path + "predicted_duty_" + ci + str_constr)


" ---------------------------------------------------------- "


def cand_prop_viz_pareto_d(params, res_main_path, path, model_spec):
    plt.clf()
    y_scale = 1000
    l, c, ci, solvent = model_spec

    if params["task"] == "solvent_process":
        if ci == "T1T2":
            xlim_dict = {
                "0_C5H8O2-D1": (0.92, 1),
                "1_C4H7NO-E2": (0.92, 1),
                "2_C4H9NO-D0": (0.92, 1),
                "3_C4H6O2-N1": (0.92, 1),
                "4_C6H12O3-E2": (0.92, 1),
                "5_C5H8O3-D2": (0.92, 1),
                "6_C4H6O3": (0.92, 1),
                "7_C8H8O-D3": (0.92, 1),
                "8_C4H10N2": (0.8, 1)
            }
            ylim = extend_xylim(20000 / y_scale, 40000 / y_scale)
            y_mean, y_std = 3883.816608040528 + 3090.2694132801953, np.sqrt(
                1762.6347594241297 ** 2 + 1044.3813734345836 ** 2)
    dot_color = "blue" if params["task"] == "solvent" else "tab:grey"

    str_constr = "" if c is None else "_constr"
    res_path = res_main_path + "/optimization_" + ci + str_constr + "/"

    if params["task"] == "solvent_process":
        df_cand = pd.read_csv(res_path + f"process_evaluation_{solvent}.csv")
        if ci == "T1T2":
            df_cand = df_cand[df_cand["hasERROR"] == False]
            x_cand, y_cand = df_cand["f_ave_DIST_C4H8_T1"].values, \
                             (df_cand["f_ave_RebDuty_T1"].values + df_cand["f_ave_RebDuty_T2"].values) / y_scale
            x_true, y_true = df_cand["DIST_C4H8_T1"].values, df_cand["RebDuty_total"].values / y_scale
        xticks = None

    print(f"# of data points: {len(x_cand)}", flush=True)
    plt.scatter(x_cand, y_cand, c=dot_color, s=dot_size, zorder=1, label="Model-based estimation")
    plt.axhline(y=0, c="k", linestyle="--", zorder=0)
    plt.axvline(x=1, c="k", linestyle="--", zorder=0)
    plt.plot(1, 0, "*", c="red", ms=10)

    if params["task"] == "solvent":
        F = np.load(res_path + "F_best.npy")
        F = np.array([[1 - f[0], f[1] * y_std + y_mean] for f in F])
        x_pareto, y_pareto = F[:, 0], F[:, 1] / y_scale
        plt.scatter(x_pareto, y_pareto, c="tab:grey", s=dot_size, zorder=0)

    # if "T2" not in ci: plt.scatter(x_pred, y_pred, c="limegreen", s=dot_size, zorder=5, label="Benchmark"), \
    #                    print(x_pred, y_pred, flush=True)
    plt.xlabel("Predicted purity", size=label_size)
    plt.ylabel("Predicted heat duty (MW)", size=label_size)
    plt.xlim(xlim_dict[solvent])
    plt.ylim(ylim)
    # plt.legend(prop={"size": legend_size})
    if params["task"] == "solvent":
        if xticks is not None: plt.xticks(xticks)
        plt.savefig(path + l + "_" + ci + "_pred")
    else:
        plt.scatter(x_true, y_true, c="blue", s=dot_size, label="Process simulation")
        # plt.legend(prop={"size: legend_size})
        # if "T2" not in ci: plt.scatter(x, y, c="limegreen", s=dot_size, zorder=5, label="Benchmark"), \
        #                    print(x, y, flush=True)

        plt.xlabel("C$_4$H$_8$ purity", size=label_size)
        plt.ylabel("Total heat duty (MW)", size=label_size)
        plt.xlim(xlim_dict[solvent])
        plt.ylim(ylim)
        plt.legend(prop={"size": legend_size})
        plt.savefig(path + l + "_" + ci + str_constr + "_comp")

    x_low, x_up = xlim_dict[solvent]
    y_low, y_up = ylim
    if mode == "doc":
        x_text_1, y_text_1 = x_low + 0.1 * (x_up - x_low), x_low + 0.85 * (x_up - x_low)
        x_text_2, y_text_2 = y_low + 0.1 * (y_up - y_low), y_low + 0.85 * (y_up - y_low)
    elif mode == "ppt":
        x_text_1, y_text_1 = x_low + 0.1 * (x_up - x_low), x_low + 0.85 * (x_up - x_low)
        x_text_2, y_text_2 = y_low + 0.1 * (y_up - y_low), y_low + 0.85 * (y_up - y_low)

    plt.clf()
    plt.scatter(x_true, x_cand, c="blue", s=dot_size)
    # if "T2" not in ci: plt.scatter(x, x_pred, c="limegreen", s=dot_size, zorder=1)
    plt.xlabel("C$_4$H$_8$ purity", size=label_size)
    plt.ylabel("Predicted purity", size=label_size)
    plt.xlim(xlim_dict[solvent])
    plt.ylim(xlim_dict[solvent])
    if xticks is not None: plt.xticks(xticks)
    plt.plot([-1, 2], [-1, 2], color="k", zorder=0)
    s1 = "MAE = " + str("{:.4f}".format(mean_absolute_error(x_true, x_cand)))
    plt.text(x_text_1, y_text_1, s1, c="k", fontsize=text_size)
    # plt.scatter(y_true, y_cand)
    plt.savefig(path + "predicted_purity_" + ci + str_constr)

    plt.clf()
    plt.scatter(y_true, y_cand, c="blue", s=dot_size)
    # if "T2" not in ci: plt.scatter(y, y_pred, c="limegreen", s=dot_size, zorder=1)
    plt.xlabel("Reboiler heat duty (MW)", size=label_size)
    plt.ylabel("Predicted heat duty (MW)", size=label_size)
    plt.xlim(ylim)
    plt.ylim(ylim)
    # plt.xticks(np.arange(1, 9))
    if xticks is not None: plt.xticks(xticks)
    plt.plot([-10, 100], [-10, 100], color="k", zorder=0)
    s2 = "MAE = " + str("{:.4f}".format(mean_absolute_error(y_true, y_cand))) + " MW"
    plt.text(x_text_2, y_text_2, s2, c="k", fontsize=text_size)

    # plt.scatter(y_true, y_cand)

    plt.savefig(path + "predicted_duty_" + ci + str_constr)


" ---------------------------------------------------------- "


def Figure_3a(res_main_path):
    """ # Visualize candidates' performance """
    print("Figure 3a")
    params = get_param("solvent")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/3_cand_prop_pareto" + "/")
        cand_prop_viz_pareto_a(params, res_main_path, sub_viz_path, (l, c, "T1"))
    plt.close()


def Figure_3b(res_main_path, ci):
    """ # Visualize candidates' performance """
    print("Figure 3b:", ci)
    params = get_param("process")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/3_cand_prop_pareto/")
        cand_prop_viz_pareto_a(params, res_main_path, sub_viz_path, (l, c, ci))
    plt.close()


def Figure_3c(res_main_path, ci):
    """ # Visualize candidates' performance """
    print("Figure 3c:", ci)
    params = get_param("solvent_process")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/3_cand_prop_pareto/")
        cand_prop_viz_pareto_c(params, res_main_path, sub_viz_path, (l, c, ci))
    plt.close()


def Figure_3d(res_main_path, ci):
    """ # Visualize candidates' performance """
    print("Figure 3d:", ci)
    params = get_param("solvent_process")
    solvents = ["0_C5H8O2-D1", "1_C4H7NO-E2", "3_C4H6O2-N1", "5_C5H8O3-D2", "6_C4H6O3", "7_C8H8O-D3", "8_C4H10N2"]
    for solvent in solvents:
        for l, c in itertools.product(["nonlinear"], [None]):
            sub_viz_path = create_directory(params["viz_path"] + mode + f"/3_cand_prop_pareto_{solvent}/")
            cand_prop_viz_pareto_d(params, res_main_path, sub_viz_path, (l, c, ci, solvent))
    plt.close()


" #################################################################################################################### "


def cand_prop_viz_a(params, res_main_path, path, model_spec):
    y_scale = 1000
    l, c, ci = model_spec

    if params["task"] == "solvent":
        xlim_dict = {"nonlinear": extend_xylim(0.9, 1)}
        ylim_dict = {"nonlinear": extend_xylim(0000 / y_scale, 50000 / y_scale)}
        x, y = 0.995466491, 16660.6957 / y_scale

    elif params["task"] == "process":
        if ci == "T1":
            xlim_dict = {"nonlinear": extend_xylim(0.9, 1)}
            ylim_dict = {"nonlinear": extend_xylim(1800 / y_scale, 3200 / y_scale)}
            x, y = 0.992693432, 2596.52976 / y_scale
        elif ci == "T2":
            xlim_dict = {"nonlinear": extend_xylim(0.99, 1)}
            ylim_dict = {"nonlinear": extend_xylim(2400 / y_scale, 3000 / y_scale)}
        elif ci == "T1T2":
            xlim_dict = {"nonlinear": extend_xylim(0.9, 1)}
            ylim_dict = {"nonlinear": extend_xylim(3000 / y_scale, 7000 / y_scale)}

    dot_color = "blue" if params["task"] == "solvent" else "blue"
    plt.clf()

    # plt.fill([x, x, 2, 2], [y, 10 * y, 10 * y, y], c="tab:green", alpha=0.2)
    if "T2" not in ci:
        plt.fill([x, x, 1, 1], [0, y, y, 0], c="limegreen", alpha=0.3, edgecolor="k", hatch="////", zorder=0)
        plt.axvline(x=x, c="limegreen", linestyle="--", alpha=1, zorder=1)
        plt.axhline(y=y, c="limegreen", linestyle="--", alpha=1, zorder=1)
        plt.scatter(x, y, c="limegreen", s=dot_size, zorder=2, label="Benchmark")
    plt.axhline(y=0, c="k", linestyle="--", alpha=1, zorder=1)
    plt.axvline(x=1, c="k", linestyle="--", alpha=1, zorder=1)

    str_constr = "" if c is None else "_constr"
    res_path = res_main_path + "/optimization_" + ci + str_constr + "/"

    if params["task"] == "solvent":
        df_cand = pd.read_csv(res_path + "identified_candidate.csv")
        df_cand = df_cand[df_cand["DIST_C4H8"] >= 0.5]
        x_true, y_true = df_cand["DIST_C4H8"].values, df_cand["RebDuty"].values / y_scale
        xticks = np.arange(0.9, 1.01, 0.02)
    elif params["task"] == "process":
        df_cand = pd.read_csv(res_path + "process_evaluation_" + ci + ".csv")
        if ci == "T1":
            df_cand = df_cand[df_cand["hasERROR_" + ci] == False]
            x_cand, y_cand = df_cand["f_ave_C4H8"].values, df_cand["f_ave_Duty"].values / y_scale
            x_true, y_true = df_cand["DIST_C4H8_" + ci].values, df_cand["RebDuty_" + ci].values / y_scale
        elif ci == "T2":
            df_cand = df_cand[df_cand["hasERROR_" + ci] == False]
            x_cand, y_cand = df_cand["f_ave_C4H6"].values, df_cand["f_ave_Duty"].values / y_scale
            x_true, y_true = df_cand["DIST_C4H6_" + ci].values, df_cand["RebDuty_" + ci].values / y_scale
        elif ci == "T1T2":
            df_cand = df_cand[df_cand["hasERROR"] == False]
            x_cand, y_cand = df_cand["f_ave_DIST_C4H8_T1"].values, \
                             (df_cand["f_ave_RebDuty_T1"].values + df_cand["f_ave_RebDuty_T2"].values) / y_scale
            x_true, y_true = df_cand["DIST_C4H8_T1"].values, df_cand["RebDuty_total"].values / y_scale
        xticks = None

    print(f"# of data points: {len(df_cand)}", flush=True)
    # plt.scatter(x_cand, y_cand, edgecolors="tab:grey", facecolors="none", s=dot_size, zorder=2)
    plt.scatter(x_true, y_true, c=dot_color, s=dot_size, zorder=3, label="Solvent candidate")
    plt.plot(1, 0, "*", c="red", ms=10)
    if ci == "T1":
        plt.xlabel("C$_4$H$_8$ purity", size=label_size)
    elif ci == "T2":
        plt.xlabel("C$_4$H$_6$ purity", size=label_size)
    plt.ylabel("Reboiler heat duty (MW)", size=label_size)
    plt.xlim(xlim_dict[l])
    plt.ylim(ylim_dict[l])
    if xticks is not None: plt.xticks(xticks)
    if params["task"] != "solvent": plt.legend(prop={"size": legend_size})
    plt.savefig(path + l + "_" + ci + "_simu")


def cand_prop_viz_c(params, path):
    y_scale = 1000

    alia = "C5H9NO-D2"
    name = "N-methyl-2-pyrrolidone"
    x, y = 0.995556453, 28722.8724 / y_scale

    alias = ["0_C5H8O2-D1", "1_C4H7NO-E2", "2_C4H9NO-D0",
             "3_C4H6O2-N1", "4_C6H12O3-E2", "5_C5H8O3-D2",
             "6_C4H6O3", "7_C8H8O-D3", "8_C4H10N2"]
    # names = ["Acetylacetone", "3-Methoxypropionitrile", "na",
    #          "2,3-Butanedione", "na", "Methyl acetoacetate",
    #          "Acetic anhydride", "3-Methylbenzaldehyde", "Piperazine"]
    purities = [0.996461199, 0.992114295, 0.970454812,
                0.986881101, 0.92657804, 0.994260336,
                0.995181747, 0.970643601, 0.893525629]
    duties = np.array([27569.16833, 32308.5772, 31153.1198,
                       25765.28744, 34093.2024, 33741.9732,
                       26761.06467, 32168.2182, 26864.5371]) / y_scale

    plt.fill([0.995, 0.995, 1, 1], [0, 50000, 50000, 0], c="limegreen", alpha=0.3, zorder=0)
    # plt.axvline(x=0.995, c="limegreen", linestyle="--", alpha=1, zorder=1)
    # plt.axhline(y=y, c="limegreen", linestyle="--", alpha=1, zorder=0)
    plt.scatter(x, y, c="limegreen", s=dot_size, zorder=2, label="Benchmark")
    # plt.axhline(y=0, c="k", linestyle="--", alpha=1, zorder=0)
    # plt.axvline(x=1, c="k", linestyle="--", alpha=1, zorder=1)

    plt.scatter(purities, duties, c="b", s=dot_size, zorder=3, label="Solvent candidate")
    plt.plot(1, 0, "*", c="red", ms=10)
    plt.xlabel("C$_4$H$_8$ purity", size=label_size)
    plt.ylabel("Total heat duty (MW)", size=label_size)
    plt.xlim((0.88, 1))
    plt.ylim(extend_xylim(20000 / y_scale, 40000 / y_scale))
    plt.yticks(np.arange(20, 41, 5))
    if params["task"] != "solvent": plt.legend(prop={"size": legend_size})
    plt.savefig(path + "simu")


" ---------------------------------------------------------- "


def Figure_4a(res_main_path):
    """ # Visualize candidates' performance """
    print("Figure 4a:")
    params = get_param("solvent")
    for l, c in itertools.product(["nonlinear"], [None]):
        sub_viz_path = create_directory(params["viz_path"] + mode + "/4_cand_prop/")
        cand_prop_viz_a(params, res_main_path, sub_viz_path, (l, c, "T1"))
    plt.close()


# def Figure_4b(res_main_path, ci):
#     """ # Visualize candidates' performance """
#     print("Figure 4b:", ci)
#     params = get_param("process")
#     for l, w, c in itertools.product(["nonlinear"], ["w_weights"], [None]):
#         sub_viz_path = create_directory(params["viz_path"] + "4_cand_prop/")
#         cand_prop_viz_a(params, res_main_path, sub_viz_path, (l, w, c, ci))
#     plt.close()


def Figure_4c():
    """ # Visualize candidates' performance """
    print("Figure 4c:")
    params = get_param("solvent_process")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/4_cand_prop/")
    cand_prop_viz_c(params, sub_viz_path)
    plt.close()


" ####################################################################################################################  "


def statistical_viz(params, df, path):
    plt.rcParams["figure.figsize"] = (0.5, 8)
    label_dict = {"S_Aspen": "Selectivity",
                  "MW_Aspen": "Molecular weight (g/mol)",
                  "RHO_Aspen": "Density (kg/m$^3$)",
                  "CPMX_Aspen": "Heat capacity (kJ/kmol K)",
                  "MUMX_Aspen": "Viscosity (cP)",
                  "DIST_C4H8": "C$_4$H$_8$ purity",
                  "RebDuty": "Reboiler heat duty (MW)"}
    mean_dict = {"S_Aspen": 1.0750789245099999,
                 "MW_Aspen": 120.23537299999991,
                 "RHO_Aspen": 883.7173103400003,
                 "CPMX_Aspen": 221.0291599500001,
                 "MUMX_Aspen": 1.1027967863600003,
                 "DIST_C4H8": 0,
                 "RebDuty": 6424.4099494}
    std_dict = {"S_Aspen": np.sqrt(0.021220593697380444),
                "MW_Aspen": np.sqrt(471.01573975057096),
                "RHO_Aspen": np.sqrt(21428.091385087053),
                "CPMX_Aspen": np.sqrt(2515.0418696275606),
                "MUMX_Aspen": np.sqrt(1.6028674209045661),
                "DIST_C4H8": np.sqrt(1),
                "RebDuty": np.sqrt(2932051.056743796)}

    df_NMP = df[df["alias"] == "C5H9NO-D2"]
    df_other = df[df["alias"] != "C5H9NO-D2"]

    for i, column in enumerate(list(label_dict.keys())):
        plt.clf()
        # x = df[column].values
        x_NMP, x_other = df_NMP[column].values, df_other[column].values
        y_NMP, y_other = np.zeros_like(x_NMP), np.zeros_like(x_other)
        plt.plot(y_other, x_other, "o", c="blue")
        plt.plot(y_NMP, x_NMP, "o", c="limegreen")
        color_list = ["red", "orangered", "coral", "tomato"]
        if column not in ["DIST_C4H8", "RebDuty"]:
            plt.plot(0, mean_dict[column], "_", c="blue", ms=20, markeredgewidth=1.5)
            for j in range(4):
                plt.plot(0, mean_dict[column] + (j + 1) * std_dict[column], "_", c=color_list[j], ms=20,
                         markeredgewidth=1.5, alpha=1, zorder=0)
                plt.plot(0, mean_dict[column] - (j + 1) * std_dict[column], "_", c=color_list[j], ms=20,
                         markeredgewidth=1.5, alpha=1, zorder=0)

        plt.ylabel(label_dict[column], size=label_size)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.savefig(path + str(i) + "_" + column + "_v")


def Figure_xa():
    params = get_param("solvent")
    df = pd.read_csv("../data/solvent/candidate_AorB_303.csv")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/statistics/")
    df = clean_data(params, df)
    statistical_viz(params, df, sub_viz_path)


" #################################################################################################################### "


def sol_distribution_a(params, res_main_path, path, model_spec):
    #
    # if mode == "doc":
    #     plt.rcParams["font.size"] = "9"
    #     label_size = 9.5
    #     text_size = 8.5
    #     legend_size = 8
    #     dot_size = 20
    # elif mode == "ppt":
    plt.rcParams["font.size"] = "14"
    label_size = 14.5
    text_size = 10
    legend_size = 9.5
    dot_size = 50
    #

    plt.clf()
    ci, solvents = model_spec

    if params["task"] == "solvent":
        Column_name = ["Selectivity", "Molecular weight (g/mol)", "Density (kg/m$^3$)",
                       "Molar specific heat (kJ/kmol-K)", "Viscosity (cP)"]
        df = pd.read_csv(res_main_path + "/RebDuty_T1/used_data.csv")
    elif params["task"] == "solvent_process":
        Column_name = ["Selectivity", "Relative volatility", "Molecular weight (g/mol)",
                       "Density (kg/m$^3$)", "Heat capacity (kJ/kmol-K)", "Viscosity (cP)"]
        df = pd.read_csv(params["main_in_path"] + "solvent_list.csv")

    df_NMP = df[df["alias"] == "C5H9NO-D2"]  # "C2H3N" "C3H7NO"

    res_path = res_main_path + "/optimization_" + ci + "/"
    df_map = pd.read_csv(res_path + "data_for_mapping.csv")
    # df_alt = df_map.apply(lambda row: row[df_map["alias"].isin(["C2H3N", "C3H7NO"])])
    df_cand = df_map.apply(lambda row: row[df_map["alias"].isin(solvents)])

    Columns = params["sol_label"]
    n_col = len(Columns)

    fig, axes = plt.subplots(len(Columns), len(Columns), figsize=(3.3 * n_col, 3 * n_col), tight_layout=True)

    for i in range(n_col):
        for j in range(n_col):
            # If this is the lower-triangule, add a scatterlpot for each group.
            if i > j:
                axes[i, j].scatter(Columns[j], Columns[i], c="silver", s=dot_size, alpha=0.7, linewidths=0, data=df)
                axes[i, j].scatter(Columns[j], Columns[i], c="limegreen", s=dot_size, alpha=0.7, linewidths=0,
                                   data=df_NMP)
                # axes[i, j].scatter(Columns[j], Columns[i], c="k", s=dot_size, alpha=0.7, linewidths=0, data=df_alt)
                axes[i, j].scatter(Columns[j], Columns[i], c="b", s=dot_size, alpha=0.7, linewidths=0, data=df_cand)

            if i == j:
                axes[i, j].hist(Columns[j], bins=15, alpha=0.7, data=df)

            if i == n_col - 1: axes[i, j].set_xlabel(Column_name[j], size=label_size)
            if j == 0: axes[i, j].set_ylabel(Column_name[i], size=label_size)

    for i in range(n_col):
        for j in range(n_col):
            if i < j:
                axes[i, j].remove()

    plt.savefig(path + "sol_dist")
    # plt.show()


def sol_distribution_d(params, res_main_path, path, model_spec):
    #
    # if mode == "doc":
    #     plt.rcParams["font.size"] = "9"
    #     label_size = 9.5
    #     text_size = 8.5
    #     legend_size = 8
    #     dot_size = 20
    # elif mode == "ppt":
    plt.rcParams["font.size"] = "14"
    label_size = 14.5
    text_size = 10
    legend_size = 9.5
    dot_size = 50
    #

    plt.clf()
    ci, solvents, bad_sols = model_spec

    xylim_dict = [extend_xylim(0.75, 1.75), extend_xylim(1, 6), extend_xylim(50, 200),
                  extend_xylim(600, 1200), extend_xylim(100, 400), extend_xylim(0, 3)]
    xystick_dict = [[0.75, 1.0, 1.25, 1.5, 1.75], [1, 2, 3, 4, 5, 6], None,
                    None, None, None]

    Column_name = ["Selectivity", "Relative volatility", "Molecular weight (g/mol)",
                   "Density (kg/m$^3$)", "Heat capacity (kJ/kmol-K)", "Viscosity (cP)"]
    df = pd.read_csv(params["main_in_path"] + "solvent_list.csv")

    df_NMP = df[df["alias"] == "C5H9NO-D2"]  # "C2H3N" "C3H7NO"

    res_path = res_main_path + "/optimization_" + ci + "/"
    df_map = pd.read_csv(res_path + "data_for_mapping.csv")
    # df_alt = df_map.apply(lambda row: row[df_map["alias"].isin(["C2H3N", "C3H7NO"])])
    df_cand = df_map.apply(lambda row: row[df_map["alias"].isin(solvents)])
    df_bad = df_map.apply(lambda row: row[df_map["alias"].isin(bad_sols)])
    print(df_bad)

    Columns = params["sol_label"]
    n_col = len(Columns)

    res_path = res_main_path + "/optimization_" + ci + f"/optimization.pickle"
    print(res_path, flush=True)
    res_history = pickle_load(res_path)
    X = np.array([res.X for res in res_history[-1].opt])[:, :6]
    df_pareto = pd.DataFrame(X, columns=Columns)

    fig, axes = plt.subplots(len(Columns), len(Columns), figsize=(3.3 * n_col, 3 * n_col), tight_layout=True)

    for i in range(n_col):
        for j in range(n_col):
            # If this is the lower-triangule, add a scatterlpot for each group.
            if (i >= j) and (i != n_col - 1):
                axes[i, j].scatter(Columns[j], Columns[i + 1], c="silver", s=dot_size, alpha=0.7, linewidths=0, data=df)
                axes[i, j].scatter(Columns[j], Columns[i + 1], c="limegreen", s=dot_size, alpha=0.7, linewidths=0,
                                   data=df_NMP)
                axes[i, j].scatter(Columns[j], Columns[i + 1], c="c", s=dot_size, alpha=0.7, linewidths=0,
                                   data=df_pareto)
                axes[i, j].scatter(Columns[j], Columns[i + 1], c="b", s=dot_size, alpha=0.7, linewidths=0, data=df_cand)
                axes[i, j].scatter(Columns[j], Columns[i + 1], c="r", s=dot_size, alpha=0.7, linewidths=0, data=df_bad)
                axes[i, j].set_xlim(xylim_dict[j])
                axes[i, j].set_ylim(xylim_dict[i + 1])
                if xystick_dict[j] is not None: axes[i, j].set_xticks(xystick_dict[j])
                if (xystick_dict[i + 1] is not None) and (i != n_col - 1): axes[i, j].set_yticks(xystick_dict[i + 1])

            if i == n_col - 1:
                axes[i, j].hist(Columns[j], bins=15, color="silver", alpha=0.7, data=df)
                axes[i, j].set_xlim(xylim_dict[j])
                if xystick_dict[j] is not None: axes[i, j].set_xticks(xystick_dict[j])
            #
            if i == n_col - 1: axes[i, j].set_xlabel(Column_name[j], size=label_size)
            if (j == 0) and (i != n_col - 1): axes[i, j].set_ylabel(Column_name[i + 1], size=label_size)
            if (j == 0) and (i == n_col - 1): axes[i, j].set_ylabel("Frequency")

    for i in range(n_col):
        for j in range(n_col):
            if i < j:
                axes[i, j].remove()

    plt.savefig(path + "sol_dist_pareto")
    # plt.show()

    # plt.clf()
    # fig, axes = plt.subplots(1, len(Columns), figsize=(3.3 * n_col, 3), tight_layout=True)
    # print(axes)
    # for i in range(n_col):
    #     axes[i].hist(Columns[i], bins=15, color="silver", alpha=0.7, data=df)
    #     axes[i].set_xlim(xylim_dict[i])
    #     if xystick_dict[i] is not None: axes[i].set_xticks(xystick_dict[i])
    #
    #     axes[i].set_xlabel(Column_name[i], size=label_size)
    #     if i == 0: axes[i].set_ylabel(Column_name[i], size=label_size)
    # plt.savefig(path + "sol_dist_hist")


" ---------------------------------------------------------- "


def Figure_5a(res_main_path):
    """ Visualize training samples and optimally design samples in 2D space """
    print("Figure 5a")
    params = get_param("solvent")
    solvents = ["C4H6O3", "C5H7NO2"]
    sub_viz_path = create_directory(params["viz_path"] + mode + "/5_sol_dist/")
    sol_distribution_a(params, res_main_path, sub_viz_path, ("T1", solvents))
    plt.close()


def Figure_5c(res_main_path):
    """ Visualize training samples and optimally design samples in 2D space """
    print("Figure 5c")
    params = get_param("solvent_process")
    solvents = ["C5H8O2-D1", "C4H5N-1", "C4H6O3"]
    sub_viz_path = create_directory(params["viz_path"] + mode + "/5_sol_dist/")
    sol_distribution_a(params, res_main_path, sub_viz_path, ("T1T2", solvents))
    plt.close()


def Figure_5d(res_main_path):
    """ Visualize training samples and optimally design samples in 2D space """
    print("Figure 5d")
    params = get_param("solvent_process")
    solvents = ["C5H8O2-D1", "C4H5N-1", "C4H6O3"]
    bad_sols = ["C4H6O2-N1", "C4H7NO-E2", "C4H8O2-D7", "C3H4O2-2", "C4H9NO-D0"]
    sub_viz_path = create_directory(params["viz_path"] + mode + "/5_sol_dist/")
    sol_distribution_d(params, res_main_path, sub_viz_path, ("T1T2", solvents, bad_sols))
    plt.close()


" #################################################################################################################### "


def TAC_pie(path):
    if mode == "doc":
        plt.rcParams["font.size"] = "9"
        label_size = 9.5
        text_size = 8.5
        legend_size = 8
        dot_size = 20
    elif mode == "ppt":
        plt.rcParams["font.size"] = "9"
        label_size = 11.5
        text_size = 11
        legend_size = 9
        dot_size = 30

    i = 0.1
    n = 3
    F = i * ((1 + i) ** n) / (((1 + i) ** n) - 1)
    labels = ["CAPEX$_{EDC}$", "CAPEX$_{SRC}$", "OPEX$_{EDC}$", "OPEX$_{SRC}$"]

    # 0: NMP
    plt.clf()
    CS1, CS2, CT1, CT2 = 2678402.37, 460292.457, 212791.53, 23123.6563
    HER1, HER2, HEC1, HEC2 = 1040191.15, 380548.835, 982931.09, 353517.195,
    UCR1, UCR2, UCC1, UCC2 = 2205869.2, 3284416.32, 1328214.16, 311866.606
    CAPEX_1 = F * (CS1 + CT1 + HER1 + HEC1)
    CAPEX_2 = F * (CS2 + CT2 + HER2 + HEC2)
    OPEX_1 = UCR1 + UCC1
    OPEX_2 = UCR2 + UCC2
    TAC_NMP = CAPEX_1 + CAPEX_2 + OPEX_1 + OPEX_2
    print(CAPEX_1, CAPEX_2, OPEX_1, OPEX_2, TAC_NMP)

    x = np.array([CAPEX_1, CAPEX_2, OPEX_1, OPEX_2]) / TAC_NMP
    patches, texts, autotexts = plt.pie(x, autopct="%1.2f%%", startangle=90, pctdistance=0.8, normalize=False,
                                        wedgeprops={"linewidth": 1, "edgecolor": "w", "alpha": 1})
    hole = plt.Circle((0, 0), 0.6, facecolor="white")
    plt.gcf().gca().add_artist(hole)
    plt.text(0, 0, str(np.round(TAC_NMP / 10 ** 6, 2)) + " MM$/yr", fontsize=text_size,
             horizontalalignment="center", verticalalignment="center")
    # plt.legend(patches, labels, loc="upper left", bbox_to_anchor=(1, 1), prop={"size": legend_size})
    # plt.savefig(path + "TAC_NMP")

    # 1: Acetylacetone, C5H8O2
    plt.clf()
    CS1, CS2, CT1, CT2 = 3172698.96, 295564.361, 272793.457, 13218.5477
    HER1, HER2, HEC1, HEC2 = 697562.62, 506554.067, 1263423, 456516.128
    UCR1, UCR2, UCC1, UCC2 = 2388174.49, 1516667.32, 1956943.31, 462240.314,
    CAPEX_1 = F * (CS1 + CT1 + HER1 + HEC1)
    CAPEX_2 = F * (CS2 + CT2 + HER2 + HEC2)
    OPEX_1 = UCR1 + UCC1
    OPEX_2 = UCR2 + UCC2
    TAC = CAPEX_1 + CAPEX_2 + OPEX_1 + OPEX_2
    print(CAPEX_1, CAPEX_2, OPEX_1, OPEX_2, TAC)

    x = np.array([CAPEX_1, CAPEX_2, OPEX_1, OPEX_2]) / TAC
    patches, texts, autotexts = plt.pie(x, radius=TAC / TAC_NMP, autopct="%1.2f%%", startangle=90, pctdistance=0.9,
                                        normalize=False, wedgeprops={"linewidth": 1, "edgecolor": "w", "alpha": 1})
    hole = plt.Circle((0, 0), 0.6, facecolor="white")
    plt.gcf().gca().add_artist(hole)
    plt.text(0, 0, str(np.round(TAC / 10 ** 6, 2)) + " MM$/yr", fontsize=text_size,
             horizontalalignment="center", verticalalignment="center")
    # plt.legend(patches, labels, loc="upper left", bbox_to_anchor=(1, 1), prop={"size": legend_size})
    # plt.savefig(path + "TAC_SOL1")

    # 2: Acetic anhydride, C4H6O3
    plt.clf()
    CS1, CS2, CT1, CT2 = 2900692.32, 323058.692, 245058.453, 14341.5284
    HER1, HER2, HEC1, HEC2 = 631587.426, 465714.611, 1306871.76, 401221.443
    UCR1, UCR2, UCC1, UCC2 = 2368246.96, 1409783.54, 2065502.24, 378455.977
    CAPEX_1 = F * (CS1 + CT1 + HER1 + HEC1)
    CAPEX_2 = F * (CS2 + CT2 + HER2 + HEC2)
    OPEX_1 = UCR1 + UCC1
    OPEX_2 = UCR2 + UCC2
    TAC = CAPEX_1 + CAPEX_2 + OPEX_1 + OPEX_2
    print(CAPEX_1, CAPEX_2, OPEX_1, OPEX_2, TAC)

    x = np.array([CAPEX_1, CAPEX_2, OPEX_1, OPEX_2]) / TAC
    patches, texts, autotexts = plt.pie(x, radius=TAC / TAC_NMP, autopct="%1.2f%%", startangle=90, pctdistance=0.82,
                                        normalize=False, wedgeprops={"linewidth": 1, "edgecolor": "w", "alpha": 1})
    hole = plt.Circle((0, 0), 0.6, facecolor="white")
    plt.gcf().gca().add_artist(hole)
    plt.text(0, 0, str(np.round(TAC / 10 ** 6, 2)) + " MM$/yr", fontsize=text_size,
             horizontalalignment="center", verticalalignment="center")
    plt.legend(patches, labels, loc="upper left", bbox_to_anchor=(1, 1), prop={"size": legend_size})
    # plt.savefig(path + "TAC_SOL2")


def TAC_bar(path):
    plt.rcParams["figure.figsize"] = (4, 3)

    species = ("NMP", "Acetylacetone", "Acetic anhydride")
    weight_counts = {
        "Annualized CAPEX$_{EDC}$": np.array([1.9761192695891222, 2.1740248541531707, 2.0444360892534726]),
        "Annualized CAPEX$_{SRC}$": np.array([.48956759297048306, .51143096103465214, .48428144447927456]),
        "OPEX$_{EDC}$": np.array([3.5340833600000003, 4.345117800000001, 4.4337492]),
        "OPEX$_{SRC}$": np.array([3.596282926, 1.978907634, 1.788239517])
    }

    width = 0.7

    fig, ax = plt.subplots()
    bottom = np.zeros(3)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom, alpha=0.85)
        bottom += weight_count

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={"size": legend_size})
    plt.ylim([0, 10.5])
    plt.xlabel("Solvent", size=label_size)
    plt.ylabel("TAC (MM$/yr)", size=label_size)
    plt.savefig(path + "Stacked_bar")


" ---------------------------------------------------------- "


def Figure_6c1():
    """ Visualize the portion of TAC """
    print("Figure 6c")
    params = get_param("solvent_process")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/6_TAC/")
    TAC_bar(sub_viz_path)


def Figure_6c2():
    """ Visualize the portion of TAC """
    print("Figure 6c")
    params = get_param("solvent_process")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/6_TAC/")
    TAC_pie(sub_viz_path)


" #################################################################################################################### "


def train_inter_extra(path):
    plt.clf()
    y_scale = 1000
    x_low, x_up = -10, 200
    xlabel_dict = {"DIST_C4H8_T1": "C$_4$H$_8$ purity",
                   "DIST_C4H6_T2": "C$_4$H$_6$ purity",
                   "RebDuty_T1": "EDC heat duty (MW)",
                   "RebDuty_T2": "SRC heat duty (MW)"}
    ylabel_dict = {"DIST_C4H8_T1": "Predicted C$_4$H$_8$ purity",
                   "DIST_C4H6_T2": "Predicted C$_4$H$_6$ purity",
                   "RebDuty_T1": "Predicted EDC heat duty (MW)",
                   "RebDuty_T2": "Predicted SRC heat duty (MW)"}
    xylim_dict = {"DIST_C4H8_T1": extend_xylim(0.5, 1),
                  "DIST_C4H6_T2": extend_xylim(0.4, 1),
                  "RebDuty_T1": extend_xylim(0 / y_scale, 110000 / y_scale),
                  "RebDuty_T2": extend_xylim(0 / y_scale, 35000 / y_scale)}
    diagonal_dict = {"DIST_C4H8_T1": (0, 2), "RebDuty_T1": (-10000 / y_scale, 120000 / y_scale),
                     "DIST_C4H6_T2": (0, 2), "RebDuty_T2": (-10000 / y_scale, 40000 / y_scale)}
    unit_dict = {"DIST_C4H8_T1": "",
                 "DIST_C4H6_T2": "",
                 "RebDuty_T1": " MW",
                 "RebDuty_T2": " MW"}
    xticks_dict = {"DIST_C4H8_T1": np.arange(0.5, 1.05, 0.1),
                   "DIST_C4H6_T2": np.arange(0.4, 1.05, 0.1),
                   "RebDuty_T1": np.arange(0, 120, 20),
                   "RebDuty_T2": np.arange(0, 40, 5)}

    df_T1_train = pd.read_csv("../data/train_inter_extra/T1_Train.csv")
    df_T2_train = pd.read_csv("../data/train_inter_extra/T2_Train.csv")
    df_T1_inter = pd.read_csv("../data/train_inter_extra/T1_Inter.csv")
    df_T2_inter = pd.read_csv("../data/train_inter_extra/T2_Inter.csv")
    df_T1_extra = pd.read_csv("../data/train_inter_extra/T1_Extra.csv")
    df_T2_extra = pd.read_csv("../data/train_inter_extra/T2_Extra.csv")
    print(len(df_T1_train), len(df_T2_train), len(df_T1_inter), len(df_T2_inter), len(df_T1_extra), len(df_T2_extra))

    perf = "DIST_C4H8_T1"
    x_low, x_up = diagonal_dict[perf]
    xy_low, xy_up = xylim_dict[perf]
    plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
    plt.scatter(df_T1_train[perf].values, df_T1_train["Puri_pred"].values,
                c="red", s=dot_size, alpha=0.7, linewidths=0, label="Training data")
    plt.scatter(df_T1_inter[perf].values, df_T1_inter["Puri_pred"].values,
                c="blue", s=dot_size, alpha=0.7, linewidths=0, label="Interpolation")
    plt.scatter(df_T1_extra[perf].values, df_T1_extra["Puri_pred"].values,
                c="green", s=dot_size, alpha=0.7, linewidths=0, label="Extrapolation")
    plt.xlabel(xlabel_dict[perf], size=label_size)
    plt.ylabel(ylabel_dict[perf], size=label_size)
    plt.xlim(xy_low, xy_up)
    plt.ylim(xy_low, xy_up)
    if xticks_dict[perf] is not None:
        plt.xticks(xticks_dict[perf])
        plt.yticks(xticks_dict[perf])
    plt.legend(loc="lower right", prop={"size": legend_size})
    plt.savefig(path + "T1_Purity")
    print(r2_score(df_T1_train[perf].values, df_T1_train["Puri_pred"].values),
          r2_score(df_T1_inter[perf].values, df_T1_inter["Puri_pred"].values),
          r2_score(df_T1_extra[perf].values, df_T1_extra["Puri_pred"].values))
    print(mean_absolute_error(df_T1_train[perf].values, df_T1_train["Puri_pred"].values),
          mean_absolute_error(df_T1_inter[perf].values, df_T1_inter["Puri_pred"].values),
          mean_absolute_error(df_T1_extra[perf].values, df_T1_extra["Puri_pred"].values))

    plt.clf()
    perf = "RebDuty_T1"
    x_low, x_up = diagonal_dict[perf]
    xy_low, xy_up = xylim_dict[perf]
    plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
    plt.scatter(df_T1_train[perf].values / y_scale, df_T1_train["Duty_pred"].values / y_scale,
                c="red", s=dot_size, alpha=0.7, linewidths=0, label="Training data")
    plt.scatter(df_T1_inter[perf].values / y_scale, df_T1_inter["Duty_pred"].values / y_scale,
                c="blue", s=dot_size, alpha=0.7, linewidths=0, label="Interpolation")
    plt.scatter(df_T1_extra[perf].values / y_scale, df_T1_extra["Duty_pred"].values / y_scale,
                c="green", s=dot_size, alpha=0.7, linewidths=0, label="Extrapolation")
    plt.xlabel(xlabel_dict[perf], size=label_size)
    plt.ylabel(ylabel_dict[perf], size=label_size)
    plt.xlim(xy_low, xy_up)
    plt.ylim(xy_low, xy_up)
    if xticks_dict[perf] is not None:
        plt.xticks(xticks_dict[perf])
        plt.yticks(xticks_dict[perf])
    plt.legend(loc="lower right", prop={"size": legend_size})
    plt.savefig(path + "T1_Duty")
    print(r2_score(df_T1_train[perf].values / y_scale, df_T1_train["Duty_pred"].values / y_scale),
          r2_score(df_T1_inter[perf].values / y_scale, df_T1_inter["Duty_pred"].values / y_scale),
          r2_score(df_T1_extra[perf].values / y_scale, df_T1_extra["Duty_pred"].values / y_scale))
    print(mean_absolute_error(df_T1_train[perf].values / y_scale, df_T1_train["Duty_pred"].values / y_scale),
          mean_absolute_error(df_T1_inter[perf].values / y_scale, df_T1_inter["Duty_pred"].values / y_scale),
          mean_absolute_error(df_T1_extra[perf].values / y_scale, df_T1_extra["Duty_pred"].values / y_scale))

    plt.clf()
    perf = "DIST_C4H6_T2"
    x_low, x_up = diagonal_dict[perf]
    xy_low, xy_up = xylim_dict[perf]
    plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
    plt.scatter(df_T2_train[perf].values, df_T2_train["Puri_pred"].values,
                c="red", s=dot_size, alpha=0.7, linewidths=0, label="Training data")
    plt.scatter(df_T2_inter[perf].values, df_T2_inter["Puri_pred"].values,
                c="blue", s=dot_size, alpha=0.7, linewidths=0, label="Interpolation")
    plt.scatter(df_T2_extra[perf].values, df_T2_extra["Puri_pred"].values,
                c="green", s=dot_size, alpha=0.7, linewidths=0, label="Extrapolation")
    plt.xlabel(xlabel_dict[perf], size=label_size)
    plt.ylabel(ylabel_dict[perf], size=label_size)
    plt.xlim(xy_low, xy_up)
    plt.ylim(xy_low, xy_up)
    if xticks_dict[perf] is not None:
        plt.xticks(xticks_dict[perf])
        plt.yticks(xticks_dict[perf])
    plt.legend(loc="lower right", prop={"size": legend_size})
    plt.savefig(path + "T2_Purity")
    print(r2_score(df_T2_train[perf].values, df_T2_train["Puri_pred"].values),
          r2_score(df_T2_inter[perf].values, df_T2_inter["Puri_pred"].values),
          r2_score(df_T2_extra[perf].values, df_T2_extra["Puri_pred"].values))
    print(mean_absolute_error(df_T2_train[perf].values, df_T2_train["Puri_pred"].values),
          mean_absolute_error(df_T2_inter[perf].values, df_T2_inter["Puri_pred"].values),
          mean_absolute_error(df_T2_extra[perf].values, df_T2_extra["Puri_pred"].values))

    plt.clf()
    perf = "RebDuty_T2"
    x_low, x_up = diagonal_dict[perf]
    xy_low, xy_up = xylim_dict[perf]
    plt.plot([x_low, x_up], [x_low, x_up], color="k", zorder=0)
    plt.scatter(df_T2_train[perf].values / y_scale, df_T2_train["Duty_pred"].values / y_scale,
                c="red", s=dot_size, alpha=0.7, linewidths=0, label="Training data")
    plt.scatter(df_T2_inter[perf].values / y_scale, df_T2_inter["Duty_pred"].values / y_scale,
                c="blue", s=dot_size, alpha=0.7, linewidths=0, label="Interpolation")
    plt.scatter(df_T2_extra[perf].values / y_scale, df_T2_extra["Duty_pred"].values / y_scale,
                c="green", s=dot_size, alpha=0.7, linewidths=0, label="Extrapolation")
    plt.xlabel(xlabel_dict[perf], size=label_size)
    plt.ylabel(ylabel_dict[perf], size=label_size)
    plt.xlim(xy_low, xy_up)
    plt.ylim(xy_low, xy_up)
    if xticks_dict[perf] is not None:
        plt.xticks(xticks_dict[perf])
        plt.yticks(xticks_dict[perf])
    plt.legend(loc="lower right", prop={"size": legend_size})
    plt.savefig(path + "T2_Duty")
    print(r2_score(df_T2_train[perf].values / y_scale, df_T2_train["Duty_pred"].values / y_scale),
          r2_score(df_T2_inter[perf].values / y_scale, df_T2_inter["Duty_pred"].values / y_scale),
          r2_score(df_T2_extra[perf].values / y_scale, df_T2_extra["Duty_pred"].values / y_scale))
    print(mean_absolute_error(df_T2_train[perf].values / y_scale, df_T2_train["Duty_pred"].values / y_scale),
          mean_absolute_error(df_T2_inter[perf].values / y_scale, df_T2_inter["Duty_pred"].values / y_scale),
          mean_absolute_error(df_T2_extra[perf].values / y_scale, df_T2_extra["Duty_pred"].values / y_scale))


def Figure_7():
    print("Figure 6c")
    params = get_param("solvent_process")
    sub_viz_path = create_directory(params["viz_path"] + mode + "/7_Train_Inter_Extra/")
    train_inter_extra(sub_viz_path)
