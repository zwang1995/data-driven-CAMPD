# Created at 05 Oct 2022 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Surrogate-based prediction

from basic_utils import *
from fnn_utils import *
import joblib

perf = "DIST_C4H8_T1" #DIST_C4H8_T1 #RebDuty_T1 #DIST_C4H6_T2 #RebDuty_T2
params = get_param("solvent_process", (perf, "regression", "T2"))
prop_scaler = joblib.load(params["prop_scaler_file"])
print(prop_scaler.mean_)
if "Duty" in perf: perf_scaler = joblib.load(params["perf_scaler_file"])

n_in, n_out = len(params["prop_label"]), len(params["perf_label"])
n_layer, n_hid, act = eval(read_txt(params["best_hyper_file"])[0][0])
print(f"-> Perf: {perf} / N_layer: {n_layer} / N_neuron: ({n_in}, {n_hid}, {n_out}) / Act: {act}", flush=True)
model = FNN_model(params, n_layer, (n_in, n_hid, n_out), act)
model.eval()
out_path = params["main_out_path"]
for index in range(5):
    model_para_file = "".join([out_path, perf, "/model_", str(index), ".pt"])
    model.load_state_dict(torch.load(model_para_file))
    x_0 = [[1.557227355, 1.269685142, 86.09044, 979.078611, 160.859221, 0.751024251, 80,
  6.8701610274734755, 3.754001097387407, 3.0656632991326336]]
    # 19, 1.993502145790478, 3.537553301758379, 3.0656632991326336
    x_0 = get_Tensor(prop_scaler.transform(x_0))
    if "Duty" in perf:
        print("#", perf_scaler.inverse_transform(model(x_0).detach().numpy())[0][0], flush=True)
    else:
        print("#", model(x_0).detach().numpy()[0][0], flush=True)