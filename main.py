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

# check that a valid model was input
if model not in ["RF", "LR", "GPR", "GPR_Bayesian"]:
    print("{} is not a valid model. Please choose from 'RF', 'LR', 'GPR', or 'GPR_Bayesian'.")
    quit()

# import data set
if dataset == "Diffusion":
    X_train = np.load('diffusion_data/all_x_values.npy')
    y_train = np.load('diffusion_data/all_y_values.npy')
elif dataset == "Perovskite":
    X_train = np.load('perovskite_data/all_x_values.npy')
    y_train = np.load('perovskite_data/all_y_values.npy')
    if model == "GPR" and action == "plot":
        print("GPR was not run on Perovskite because of computational constraints.")
        print("If you would like to see results for GPR with Perovskite, please change action from 'plot' to 'run'.")
        print("(But keep in mind that it may take a long time, i.e. weeks, to run.)")
        quit()
elif dataset == "Friedman":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_hypercube.npy')
elif dataset == "Friedman_0.1_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_0.1_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_0.1_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
elif dataset == "Friedman_0.2_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_0.2_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_0.2_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
elif dataset == "Friedman_0.3_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_0.3_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_0.3_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
elif dataset == "Friedman_0.4_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_0.4_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_0.4_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
elif dataset == "Friedman_0.5_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_0.5_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_0.5_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
elif dataset == "Friedman_1.0_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_1.0_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_1.0_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
elif dataset == "Friedman_2.0_Noise":
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values_2.0_noise.npy')
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_2.0_noise.npy')
    if model != "RF" and action == "plot":
        print("Noisy data was only run using the RF model for our paper.")
        print("If you would like to see results for the {} model, please change action from 'plot' to 'run'.".format(model))
        quit()
else:
    X_train, y_train, X_test, y_test = None, None, None, None
    print("Invalid dataset provided.")
    quit()

########################## Functions to save data ###########################
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


def save_syntheticdata(a, b, residuals, unscaled_model_errors, scaled_model_errors):
    combined = []
    for i in range(0, len(residuals)):
        curr = np.asarray([residuals[i], unscaled_model_errors[i], scaled_model_errors[i]])
        combined.append(curr)

    combined = np.asarray(combined)

    calibration = np.asarray([np.asarray([a, b])])

    np.savetxt("{}/residuals_and_uncertainty_estimates.csv".format(path), combined,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("{}/calibration_values.csv".format(path), calibration, header="a, b",
               delimiter=",")


###################################### Actions based on user input ###########################
# Run action for real and synthetic data
if action == "run":
    if dataset == "Diffusion" or dataset == "Perovskite":
        print("Starting run for {} on {} data.".format(model, dataset))
        # Get test data residuals and model errors
        TD = td.TestData()
        Test_residuals, Test_model_errors_unscaled, Test_model_errors_scaled, a_array, b_array = TD.get_residuals_and_model_errors_looped(
            dataset, model, X_train, y_train, model_num=trees)

        print("Summary:")
        print("a = %.3f +/- %.3f" % (np.mean(a_array), np.std(a_array)))
        print("b = %.3f +/- %.3f" % (np.mean(a_array), np.std(a_array)))

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

    else:
        print("Starting run for {} on {} data.".format(model, dataset))
        # Get CV residuals and model errors
        CVD = cvd.CVData()
        CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors(model, X_train, y_train, model_num=trees)

        # Scale residuals and model errors by data set standard deviation
        stdev = np.std(y_train)
        CV_residuals = CV_residuals / stdev
        CV_model_errors = CV_model_errors / stdev

        # Get correction factors
        CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
        a, b, r_squared = CF.nll()
        print('Correction Factors:')
        print('a: ' + str(a))
        print('b: ' + str(b))

        # Get test data residuals and model errors
        TD = td.TestData()
        Test_residuals, Test_model_errors = TD.get_residuals_and_model_errors(model, X_train, y_train, X_test, y_test,
                                                                              model_num=trees)
        # Scale by standard deviation
        Test_residuals = Test_residuals / stdev
        Test_model_errors = Test_model_errors / stdev

        # Save data from run
        save_syntheticdata(a, b, Test_residuals, Test_model_errors, Test_model_errors * a + b)

        # Make scaled and unscaled test data plots
        MP = mp.MakePlot()
        MP.make_rstat_overlay(Test_residuals, Test_model_errors, Test_model_errors * a + b,
                              "{}, {}".format(model, dataset), save=save_plot,
                              file_name='{}/rstat.png'.format(path))

        MP.make_rve_overlay(Test_residuals, Test_model_errors, Test_model_errors * a + b,
                            "{}, {}".format(model, dataset),
                            save=save_plot,
                            file_name='{}/RvE.png'.format(path))


# Plot action for real and synthetic data
if action == "plot":
    if dataset == "Diffusion" or dataset == "Perovskite":
        # Load pre-run data
        unscaled_model_errors = np.load(
            'data_for_paper_plots/{}/{}/Test/Test_model_errors_unscaled.npy'.format(dataset, model))
        scaled_model_errors = np.load(
            'data_for_paper_plots/{}/{}/Test/Test_model_errors_scaled.npy'.format(dataset, model))
        residuals = np.load('data_for_paper_plots/{}/{}/Test/Test_residuals.npy'.format(dataset, model))
        a = np.load('data_for_paper_plots/{}/{}/Test/a.npy'.format(dataset, model))
        b = np.load('data_for_paper_plots/{}/{}/Test/b.npy'.format(dataset, model))

        print("Calibration factors:")
        print("a = %.3f +/- %.3f" % (np.mean(a), np.std(a)))
        print("b = %.3f +/- %.3f" % (np.mean(b), np.std(b)))

        # save data
        save_realdata(a, b, residuals, unscaled_model_errors, scaled_model_errors)

        # Make scaled and unscaled test data plots
        MP = mp.MakePlot()
        MP.make_rstat_overlay(residuals, unscaled_model_errors, scaled_model_errors, "{}, {}".format(model, dataset),
                              save=save_plot,
                              file_name='{}/rstat.png'.format(path))

        MP.make_rve_overlay(residuals, unscaled_model_errors, scaled_model_errors, "{}, {}".format(model, dataset),
                            save=save_plot,
                            file_name='{}/RvE.png'.format(path))

    else:
        if dataset == "Friedman":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/{}/Test/Test_model_errors.npy'.format(model))
            residuals = np.load('data_for_paper_plots/Friedman_500/{}/Test/Test_residuals.npy'.format(model))
            a = np.load('data_for_paper_plots/Friedman_500/{}/Test/a.npy'.format(model))
            b = np.load('data_for_paper_plots/Friedman_500/{}/Test/b.npy'.format(model))
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_0.1_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_0.1_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_0.1_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_0.1_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_0.1_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_0.2_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_0.2_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_0.2_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_0.2_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_0.2_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_0.3_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_0.3_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_0.3_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_0.3_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_0.3_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_0.4_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_0.4_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_0.4_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_0.4_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_0.4_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_0.5_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_0.5_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_0.5_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_0.5_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_0.5_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_1.0_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_1.0_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_1.0_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_1.0_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_1.0_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        elif dataset == "Friedman_2.0_Noise":
            unscaled_model_errors = np.load(
                'data_for_paper_plots/Friedman_500/RF_2.0_noise/Test/Test_model_errors.npy')
            residuals = np.load('data_for_paper_plots/Friedman_500/RF_2.0_noise/Test/Test_residuals.npy')
            a = np.load('data_for_paper_plots/Friedman_500/RF_2.0_noise/Test/a.npy')
            b = np.load('data_for_paper_plots/Friedman_500/RF_2.0_noise/Test/b.npy')
            scaled_model_errors = unscaled_model_errors * a[0] + b[0]
        else:
            a, b, residuals, unscaled_model_errors, scaled_model_errors = None, None, None, None, None
            print("No valid dataset provided.")
            quit()

        print("Calibration factors:")
        print("a = %.3f" % a[0])
        print("b = %.3f" % b[0])

        # Save data
        save_syntheticdata(a[0], b[0], residuals, unscaled_model_errors, scaled_model_errors)

        # make plots
        MP = mp.MakePlot()
        MP.make_rstat_overlay(residuals, unscaled_model_errors, scaled_model_errors,
                              "{}, {}".format(model, dataset), save=save_plot,
                              file_name='{}/rstat.png'.format(path))

        MP.make_rve_overlay(residuals, unscaled_model_errors, scaled_model_errors,
                            "{}, {}".format(model, dataset),
                            save=save_plot,
                            file_name='{}/RvE.png'.format(path))

