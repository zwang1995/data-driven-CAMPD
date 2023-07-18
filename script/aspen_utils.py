# Created on 12 May 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Utils for Aspen Plus COM interface

import os
import time
import win32com.client as win32
from win32com.client import GetObject, Dispatch
from scipy.optimize import fsolve
import numpy as np
import multiprocessing as mp


class Aspen_Plus_Interface():
    def __init__(self):
        self.Application = Dispatch("Apwn.Document.38.0")
        print(self.Application)

    def load_bkp(self, bkp_file, visible_state=0, dialog_state=0):
        """
        Load a process via bkp file.

        :param bkp_file: location of the bkp file
        """
        self.Application.InitFromArchive2(os.path.abspath(bkp_file))
        self.Application.Visible = visible_state  # Aspen Plus user interface: invisible 0; visible 1
        self.Application.SuppressDialogs = dialog_state  # Aspen Plus dialogs: suppress 1

    def re_initialization(self):
        """
        Aspen Plus initialization / clean previous data
        """
        self.Application.Reinit()

    def run_property(self):
        """
        run Properties for next run of Simulation
        """
        self.Application.Engine.Run()

    def run_simulation(self):
        """
        run Simulation
        """
        self.Application.Engine.Run2(1)

    def check_run_completion(self, time_limit=99):
        """
        check whether the simulation completed
        """
        times = 0
        while self.Application.Engine.IsRunning == 1:
            time.sleep(1)
            times += 1
            if times >= time_limit:
                print("* Exceed time limit")
                self.Application.Engine.Stop
                break

    def check_convergency(self):
        """
        Check convergency by detecting errors in the history file
        """
        hasIssue = self.Application.Tree.FindNode(r"\Data\Results Summary\Run-Status\Output\PER_ERROR").Value
        hasIssue = np.bool(hasIssue)
        # print("Simulation convergency:", not isError)

        runID = self.Application.Tree.FindNode("\Data\Results Summary\Run-Status\Output\RUNID").Value
        his_file = "../simulation/" + runID + ".his"
        with open(his_file, "r") as f:
            # hasError = np.any(np.array([line.find("SEVERE ERROR") for line in f.readlines()]) >= 0)
            hasError = np.any(np.array([line.find("ERROR") for line in f.readlines()]) >= 0)
        # print("Simulation convergency:", not isError)

        return hasIssue, hasError

    def close_bkp(self):
        self.Application.Quit()

    def collect_stream(self):
        """
        Colloct all streams involved in the process.

        :return: a tuple of the name of all streams
        """
        streams = []
        node = self.Application.Tree.FindNode(r"\Data\Streams")
        for item in node.Elements:
            streams.append(item.Name)
        return tuple(streams)

    def collect_block(self):
        """
        Colloct all blocks involved in the process.

        :return: a tuple of the name of all blocks
        """
        blocks = []
        node = self.Application.Tree.FindNode(r"\Data\Blocks")
        for item in node.Elements:
            blocks.append(item.Name)
        return tuple(blocks)


def KillAspen():
    WMI = GetObject('winmgmts:')
    for p in WMI.ExecQuery("select * from Win32_Process where Name='AspenPlus.exe'"):
        os.system("taskkill /pid " + str(p.ProcessId))
