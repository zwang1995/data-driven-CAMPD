# Created at 08 Jun 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Automatic process simulation for the extractive distillation process

import itertools
from aspen_utils import *
from basic_utils import *


def run_Simulation(Combine_loop, txt_file, start_Index=0):
    """ # Initial history file """

    if start_Index == 0:
        with open(txt_file, "w") as f:
            f.write(" ".join(["Index",
                              "NStage_T1", "RR_T1", "TopPres_T1", "StoF_T1", "NStage_T2", "RR_T2", "TopPres_T2",
                              "DIST_C4H8_T1", "DIST_C4H6_T1", "RebDuty_T1",
                              "DIST_C4H6_T2", "DIST_SOL_T2", "RebDuty_T2",
                              "hasIssue", "hasERROR",
                              "TimeCost",
                              "\n"]))
    """ # Initial Aspen Plus """
    Aspen_Plus = Aspen_Plus_Interface()
    """ Preparation for Aspen automation """
    Aspen_Plus.load_bkp(r"../simulation/c_ExtractiveDistillation_T1T2.bkp", 0, 1)
    time.sleep(2)

    print("Initial specification ...")
    # Aspen_Plus.Application.Tree.FindNode("\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C5H9NO-D2"
    # Aspen_Plus.Application.Tree.FindNode("\Data\Components\Specifications\Input\ANAME1\SOLVENT").Value = "C4H6O3"
    # Aspen_Plus.re_initialization()
    # Aspen_Plus.run_property()
    # Aspen_Plus.check_run_completion()

    FeedRate = 500

    print("Start simulation ...")
    TotalTimeStart = time.time()
    Output_array = []
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
            print("#", run_Index, "-", Input)

            """ # T1 variables """
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\NSTAGE").Value = NStage_T1
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\BASIS_RR").Value = RR_T1
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\PRES1").Value = TopPres_T1
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = \
                StoF_T1 * FeedRate
            " Constraint: FeedStage = 0.5 * NStage "
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\FEED_STAGE\FEED").Value = \
                np.ceil(0.5 * NStage_T1)
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\FEED\Input\PRES\MIXED").Value = TopPres_T1 + 0.5
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\PRES\MIXED").Value = TopPres_T1 + 0.5

            """ # T2 variables """
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\NSTAGE").Value = NStage_T2
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\BASIS_RR").Value = RR_T2
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\PRES1").Value = TopPres_T2
            " Constraint: FeedStage = 0.5 * NStage "
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\RD\Input\FEED_STAGE\FEED2").Value = \
                np.ceil(0.5 * NStage_T2)
            Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\COMP\Input\PRES").Value = TopPres_T2 + 0.5

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
            Output = [DIST_C4H8_T1, DIST_C4H6_T1, RebDuty_T1,
                      DIST_C4H6_T2, DIST_SOLV_T2, RebDuty_T2,
                      hasIssue, hasERROR]
            # except:
            #     SimuError = True

            if not SimuError:
                print(Output)
                Output_array.append(Output)

                """ # Simulation time """
                TimeEnd = time.time()
                TimeCost = TimeEnd - TimeStart

                """ # Store data """
                with open(txt_file, "a") as f:
                    f.write(" ".join(ListValue2Str([run_Index]) +
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
    return run_Index, Output_array


def main():
    """ # Specify variable range """
    NStage_T1_list = SequenceWithEndPoint(10, 50, 5)
    RR_T1_list = SequenceWithEndPoint(1, 10, 1)
    TopPres_T1_list = SequenceWithEndPoint(2.5, 5, 1.25)
    StoF_list = SequenceWithEndPoint(1, 8, 1)
    NStage_T2_list = SequenceWithEndPoint(8, 20, 2)
    RR_T2_list = SequenceWithEndPoint(0.1, 1, 0.1)
    TopPres_T2_list = SequenceWithEndPoint(2.5, 5, 1.25)
    Combine_loop = list(itertools.product(NStage_T1_list, RR_T1_list, TopPres_T1_list, StoF_list,
                                          NStage_T2_list, RR_T2_list, TopPres_T2_list))
    total_run = len(Combine_loop)
    print(f"Possible Combinations: {total_run}")

    txt_file = "../data/process/DistillRecoveryColumn.txt"
    run_Index = 0
    while run_Index < total_run - 1:
        print("Connecting ...")
        try:
            run_Index, _ = run_Simulation(Combine_loop, txt_file, run_Index)
            print(run_Index)
            print("Deconnecting ...")
        except:
            print("Error detected ...")

        KillAspen()
        time.sleep(10)


if __name__ == "__main__":
    main()
