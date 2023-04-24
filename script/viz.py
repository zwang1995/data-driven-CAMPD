# Created at 25 Jul 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Visualization for data distribution, model performance, and optimization results

from viz_utils import *

# Note:
# * Figure_xa: graphs for "Solvent Design"
# * Figure_xb: graphs for "Process Optimization"
# * Figure_xc/d: graphs for "Integrated Solvent and Process Design"

sol_res_main_path = "../model/solvent/archive 01.30"
pro_res_main_path = "../model/process/archive 02.13"
sol_pro_res_main_path = "../model/solvent_process/archive 02.24"

" # Linear correlation and distribution between inputs and outputs "
# Figure_0a() # task = "solvent"

# Figure_0b("T1") # task = "process"
# Figure_0b("T2") # task = "process"

# Figure_0c("T1") # task = "solvent_process"
# Figure_0c("T2") # task = "solvent_process"


" # Parity plot for data-driven models "  # 1
# Figure_1a(sol_res_main_path)
# Figure_1a(sol_res_main_path, True)

# Figure_1b(pro_res_main_path, "T1")
# Figure_1b(pro_res_main_path, "T1", True)
# Figure_1b(pro_res_main_path, "T2")
# Figure_1b(pro_res_main_path, "T2", True)

# Figure_1c(sol_pro_res_main_path, "T1")
# Figure_1c(sol_pro_res_main_path, "T1", True)
# Figure_1c(sol_pro_res_main_path, "T2")
# Figure_1c(sol_pro_res_main_path, "T2", True)


" # Pareto front of multi-objective optimization "  # 2
# Figure_2a(sol_res_main_path)

# Figure_2b(pro_res_main_path, "T1T2")

# Figure_2c(sol_pro_res_main_path, "T1T2")
# Figure_2d(sol_pro_res_main_path, "T1T2")


" # Predicted purity and duty for Pareto-optimal solutions "  # 3
# Figure_3a(sol_res_main_path)

Figure_3b(pro_res_main_path, "T1T2")

# Figure_3c(sol_pro_res_main_path, "T1T2")
# Figure_3d(sol_pro_res_main_path, "T1T2")


" # Simulated purity and duty for Pareto-optimal solutions "  # 4 - only for solvent and solvent_process
# Figure_4a(sol_res_main_path)

# Figure_4c()

" # Training samples and candidates in 2D space "
# Figure_5a(sol_res_main_path) #

# Figure_5c(sol_pro_res_main_path) # candidates
# Figure_5d(sol_pro_res_main_path) # pareto front

" # TAC "
# Figure_6c1()
# Figure_6c2()

" # Statistical plot of solvent properties "
# Figure_xa()
# Figure_xb()
