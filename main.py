import numpy as np
from package import CVData as cvd
from package import CorrectionFactors as cf
from package import MakePlot as mp
from package import TestData as td
from sklearn.model_selection import train_test_split
import sys
import os
from datetime import datetime

# Check for command line input
if len(sys.argv) < 4:
    print("Need at least 3 command line arguments. (run/plot, model, and dataset)")
    quit()
action = sys.argv[1]
model = sys.argv[2]
dataset = sys.argv[3]
if len(sys.argv) >= 5:
    path = sys.argv[4]
else:
    path = ""

# Save plots
save_plot = True

# make file to save plots
now = datetime.now()
dt_string = now.strftime("%m-%d-%Y_%H.%M.%S")
directory = "{}_{}_{}_{}".format(model, dataset, action, dt_string)
path = os.path.join(path, directory)
os.mkdir(path)

# determine number of models in ensemble
if model == "GPR":
    trees = 200
else:
    trees = 500

# import data set
if dataset == "Diffusion":
    X_train = np.load('diffusion_data/all_x_values.npy')
    y_train = np.load('diffusion_data/all_y_values.npy')
elif dataset == "Perovskite":
    X_train = np.load('perovskite_data/all_x_values.npy')
    y_train = np.load('perovskite_data/all_y_values.npy')
else:
    X_train = None
    y_train = None
    print("Invalid dataset provided.")
    quit()

# Run action for real data
if action == "run":
    if dataset == "Diffusion" or dataset == "Perovskite":
        # Get test data residuals and model errors
        TD = td.TestData()
        Test_residuals, Test_model_errors_unscaled, Test_model_errors_scaled, a_array, b_array = TD.get_residuals_and_model_errors_looped(
            dataset, model, X_train, y_train, model_num=trees)
        print('a_array:')
        print(a_array)
        print('b_array:')
        print(b_array)

        # save data
        save_realdata(a_array, b_array, Test_residuals, Test_model_errors_unscaled, Test_model_errors_scaled)

        # Make scaled and unscaled test data plots
        MP = mp.MakePlot()
        # overlay plots
        MP.make_rstat_overlay(Test_residuals, Test_model_errors_unscaled, Test_model_errors_scaled,
                              "{}, {}".format(model, dataset),
                              save=save_plot,
                              file_name='{}/rstat.png'.format(path))

        MP.make_rve_overlay(Test_residuals, Test_model_errors_unscaled, Test_model_errors_scaled,
                            "{}, {}".format(model, dataset),
                            save=save_plot,
                            file_name='{}/RvE.png'.format(path))


def save_realdata(a, b, residuals, unscaled_model_errors, scaled_model_errors):
    combined = []
    for i in range(0, len(residuals)):
        curr = np.asarray([residuals[i], unscaled_model_errors[i], scaled_model_errors[i]])
        combined.append(curr)

    combined = np.asarray(combined)

    combined_calibration = []
    for i in range(0, len(a)):
        curr = np.asarray([i + 1, a[i], b[i]])
        combined_calibration.append(curr)

    combined_calibration = np.asarray(combined_calibration)

    np.savetxt("{}/residuals_and_uncertainty_estimates.csv".format(path), combined,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("{}/calibration_values.csv".format(path), combined_calibration,
               header="cv_iteration, a, b",
               delimiter=",")
