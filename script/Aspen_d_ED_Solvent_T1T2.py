# Created at 08 Jun 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Automatic process simulation for the extractive distillation process considering different solvents

import itertools
import time

from aspen_utils import *
from basic_utils import *


def run_Simulation(Solvent_list, Combine_loop, txt_file, solvent_Index=0, start_Index=0):
    """ # Initial history file """

    if (solvent_Index == 0) & (start_Index == 0):
        with open(txt_file, "w") as f:
            f.write(" ".join(["Solvent_Index", "alias", "Run_Index",
                              "NStage_T1", "RR_T1", "TopPres_T1", "StoF_T1", "NStage_T1", "RR_T1", "TopPres_T2",
                              "DIST_C4H8_T1", "DIST_C4H6_T1", "RebDuty_T1",
                              "DIST_C4H6_T2", "DIST_SOL_T2", "RebDuty_T2",
                              "hasIssue", "hasERROR", "CAPEX", "OPEX", "TAC",
                              "TimeCost",
                              "\n"]))

    """ # Initial Aspen Plus """
    Aspen_Plus = Aspen_Plus_Interface()
    """ Preparation for Aspen automation """
    Aspen_Plus.load_bkp(r"../simulation/c_ExtractiveDistillation_T1T2_TAC.bkp", 0, 1)
    time.sleep(2)

    # print("Initial specification ...")
    # Aspen_Plus.Application.Tree.FindNode("\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C4H6O3"
    # Aspen_Plus.re_initialization()
    # Aspen_Plus.run_property()
    # Aspen_Plus.check_run_completion()

    Output_array = []
    FeedRate = 500

    print("Start simulation ...")
    TotalTimeStart = time.time()

    for s_Index, solvent in enumerate(Solvent_list):
        # Combine_loop = HybridSampling(Combine_loop_, total_run, random_state=s_Index)
        # Combine_loop = RandomStaticSampling(Combine_loop, total_run, random_state=s_Index)
        if s_Index >= solvent_Index:
            # Aspen_Plus.re_initialization()
            Aspen_Plus.Application.Tree.FindNode("\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = solvent
            # Aspen_Plus.re_initialization()
            # Aspen_Plus.run_property()
            # Aspen_Plus.check_run_completion()

        for run_Index, Input in enumerate(Combine_loop):
            SimuError = False

            if run_Index >= start_Index:
                TimeStart = time.time()
                """ # Process reset """
                Aspen_Plus.re_initialization()

                """ # Assign new operating variables """
                Input = list(Input)
                NStage_T1, RR_T1, TopPres_T1, StoF_T1, NStage_T2, RR_T2, TopPres_T2 = Input
                NStage_T1, NStage_T2 = int(NStage_T1), int(NStage_T2)
                Input = [NStage_T1, RR_T1, TopPres_T1, StoF_T1, NStage_T2, RR_T2, TopPres_T2]
                print("#", s_Index, "-", solvent, "|", run_Index, "-", Input)
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\NSTAGE").Value = NStage_T1
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\BASIS_RR").Value = RR_T1
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\PRES1").Value = TopPres_T1
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = \
                    StoF_T1 * FeedRate
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\FEED_STAGE\FEED").Value = \
                    np.ceil(0.5 * NStage_T1)
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\FEED\Input\PRES\MIXED").Value = TopPres_T1 + 0.5
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\PRES\MIXED").Value = TopPres_T1 + 0.5

                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\NSTAGE").Value = NStage_T2
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\BASIS_RR").Value = RR_T2
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\PRES1").Value = TopPres_T2
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\FEED_STAGE\FEED2").Value = \
                    np.ceil(0.5 * NStage_T2)
                Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\COMP\Input\PRES").Value = TopPres_T2 + 0.5

                # For TAC calculation only
                Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Blocks\ED\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value = NStage_T1 - 1
                Aspen_Plus.Application.Tree.FindNode(
                    "\Data\Blocks\RD\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\CS-1").Value = NStage_T2 - 1
                # For TAC calculation only

                """ # Run process simulation """
                Aspen_Plus.run_simulation()
                Aspen_Plus.check_run_completion()

                """ # Collect results """
                # try:
                DIST_C4H8_T1 = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Streams\DIST1\Output\MOLEFRAC\MIXED\C4H8").Value
                DIST_C4H6_T1 = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Streams\DIST1\Output\MOLEFRAC\MIXED\C4H6").Value
                RebDuty_T1 = Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Output\REB_DUTY").Value
                DIST_C4H6_T2 = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Streams\DIST2\Output\MOLEFRAC\MIXED\C4H6").Value
                DIST_SOLV_T2 = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Streams\BOTTOM2\Output\MOLEFRAC\MIXED\SOLVENT").Value
                RebDuty_T2 = Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Output\REB_DUTY").Value
                hasIssue, hasERROR = Aspen_Plus.check_convergency()
                time.sleep(2)

                CAPEX = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\1").Value
                OPEX = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\2").Value
                TAC = Aspen_Plus.Application.Tree.FindNode(
                    r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\3").Value

                Output = [DIST_C4H8_T1, DIST_C4H6_T1, RebDuty_T1,
                          DIST_C4H6_T2, DIST_SOLV_T2, RebDuty_T2,
                          hasIssue, hasERROR, CAPEX, OPEX, TAC]
                # except:
                #     SimuError = True

                if not SimuError:
                    print(Output)
                    time.sleep(2)
                    Output_array.append(Output)

                    """ # Simulation time """
                    TimeEnd = time.time()
                    TimeCost = TimeEnd - TimeStart

                    """ # Store data """
                    with open(txt_file, "a") as f:
                        f.write(" ".join(ListValue2Str([s_Index, solvent, run_Index]) +
                                         ListValue2Str(Input) +
                                         ListValue2Str(Output) +
                                         ListValue2Str([TimeCost]) +
                                         ["\n"]))
                else:
                    break

    print("\nTerminate simulation ...")
    TotalTimeEnd = time.time()
    print(f"Simulation time:{TotalTimeEnd - TotalTimeStart: .2f} s")
    Aspen_Plus.close_bkp()
    return s_Index, run_Index, Output_array


def main():
    """ # Specify variable range """
    # NStage_T1_list = SequenceWithEndPoint(20, 100, 10)
    # RR_T1_list = SequenceWithEndPoint(1, 10, 1)
    # TopPres_T1_list = SequenceWithEndPoint(2.5, 5, 1.25)
    # StoF_list = SequenceWithEndPoint(1, 8, 1)
    # NStage_T2_list = SequenceWithEndPoint(8, 20, 2)
    # RR_T2_list = SequenceWithEndPoint(0.1, 1, 0.1)
    # TopPres_T2_list = SequenceWithEndPoint(2.5, 5, 1.25)
    # Combine_loop = list(itertools.product(NStage_T1_list, RR_T1_list, TopPres_T1_list, StoF_list,
    #                                       NStage_T2_list, RR_T2_list, TopPres_T2_list))
    # total_run = len(Combine_loop)
    # print(f"Possible Combinations: {total_run}")

    # Process optimization
    Solvent_list = ["C5H9NO-D2"]
    Combine_loop = [
        [80, 5.829299968910926, 3.500699446263485, 2.5178639696102234, 11, 0.4709906486367485, 3.500388025764173]]

    # Integrated Solvent and Process Design
    # Solvent_list = ["C5H8O2-D1"]
    # Combine_loop = [[80, 3.956578135023906, 3.6399383708475055, 3.066919139826119, 20, 1.9998486512611129, 3.533590067065156]]
    # Solvent_list = ["C4H5N-1"]
    # Combine_loop = [[80, 6.512088945208033, 3.5254755531139432, 3.6178018690749045, 19, 1.9926309583593629, 3.5196141769059284]]
    # Solvent_list = ["C4H6O3"]
    # Combine_loop = [[80, 9.970795385324887, 3.5071209478229806, 1.7566437360909586, 20, 0.7074567331006457, 3.5007232040747733]]

    total_run = len(Combine_loop)

    txt_file = "../data/solvent_process/DistillRecoveryColumn.txt"
    solvent_Index, run_Index = 0, 0
    while solvent_Index < len(Solvent_list):
        while run_Index < total_run:
            # try:
            solvent_Index, run_Index, _ = run_Simulation(Solvent_list, Combine_loop, txt_file, solvent_Index, run_Index)
            print(f" # {run_Index}")
            # except:
            #     print(" * Error Detected & Reconnecting ...")

            KillAspen()
            time.sleep(10)

            if solvent_Index == len(Solvent_list) - 1 and run_Index == total_run - 1:
                break
        if solvent_Index == len(Solvent_list) - 1 and run_Index == total_run - 1:
            break


if __name__ == "__main__":
    main()
