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
elif model == "GPR_Both":
    trees = 200
else:
    trees = 500

# check that a valid model was input
if model not in ["RF", "LR", "GPR", "GPR_Bayesian", "GPR_Both"]:
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


def save_syntheticdata_both(a_bayes, b_bayes, residuals, unscaled_model_errors_bayes, scaled_model_errors_bayes,
                            a_bootstrap, b_bootstrap, unscaled_model_errors_bootstrap, scaled_model_errors_bootstrap):
    combined_bayes = []
    combined_bootstrap = []
    for i in range(0, len(residuals)):
        curr_bayes = np.asarray([residuals[i], unscaled_model_errors_bayes[i], scaled_model_errors_bayes[i]])
        combined_bayes.append(curr_bayes)
        curr_bootstrap = np.asarray([residuals[i], unscaled_model_errors_bootstrap[i], scaled_model_errors_bootstrap[i]])
        combined_bootstrap.append(curr_bootstrap)

    combined_bayes = np.asarray(combined_bayes)
    combined_bootstrap = np.asarray(combined_bootstrap)

    calibration_bayes = np.asarray([np.asarray([a_bayes, b_bayes])])
    calibration_bootstrap = np.asarray([np.asarray([a_bootstrap, b_bootstrap])])

    np.savetxt("{}/residuals_and_uncertainty_estimates_bayes.csv".format(path), combined_bayes,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("{}/calibration_values_bayes.csv".format(path), calibration_bayes, header="a, b",
               delimiter=",")

    np.savetxt("{}/residuals_and_uncertainty_estimates_bootstrap.csv".format(path), combined_bootstrap,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("{}/calibration_values_bootstrap.csv".format(path), calibration_bootstrap, header="a, b",
               delimiter=",")

def save_realdata_both(a_bayes, b_bayes, residuals, unscaled_model_errors_bayes, scaled_model_errors_bayes,
                       a_bootstrap, b_bootstrap, unscaled_model_errors_bootstrap, scaled_model_errors_bootstrap):
    combined_bayes = []
    combined_bootstrap = []
    for i in range(0, len(residuals)):
        curr_bayes = np.asarray([residuals[i], unscaled_model_errors_bayes[i], scaled_model_errors_bayes[i]])
        combined_bayes.append(curr_bayes)
        curr_bootstrap = np.asarray([residuals[i], unscaled_model_errors_bootstrap[i], scaled_model_errors_bootstrap[i]])
        combined_bootstrap.append(curr_bootstrap)


    combined_bayes = np.asarray(combined_bayes)
    combined_bootstrap = np.asarray(combined_bootstrap)

    combined_calibration_bayes = []
    combined_calibration_bootstrap = []
    for i in range(0, len(a_bayes)):
        curr_bayes = np.asarray([i + 1, a_bayes[i], b_bayes[i]])
        combined_calibration_bayes.append(curr_bayes)
        curr_bootstrap = np.asarray([i + 1, a_bootstrap[i], b_bootstrap[i]])
        combined_calibration_bootstrap.append(curr_bootstrap)

    combined_calibration_bayes = np.asarray(combined_calibration_bayes)
    combined_calibration_bootstrap = np.asarray(combined_calibration_bootstrap)

    np.savetxt("{}/residuals_and_uncertainty_estimates_bayes.csv".format(path), combined_bayes,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("{}/calibration_values_bayes.csv".format(path), combined_calibration_bayes,
               header="cv_iteration, a, b",
               delimiter=",")

    np.savetxt("{}/residuals_and_uncertainty_estimates_bootstrap.csv".format(path), combined_bootstrap,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("{}/calibration_values_bootstrap.csv".format(path), combined_calibration_bootstrap,
               header="cv_iteration, a, b",
               delimiter=",")


###################################### Actions based on user input ###########################
# Run action for real and synthetic data
if action == "run":
    if dataset == "Diffusion" or dataset == "Perovskite":
        print("Starting run for {} on {} data.".format(model, dataset))
        # Get test data residuals and model errors
        TD = td.TestData()
        Test_residuals, Test_model_errors_unscaled_bayes, Test_model_errors_scaled_bayes, a_array_bayes, b_array_bayes, Test_model_errors_unscaled_bootstrap, Test_model_errors_scaled_bootstrap, a_array_bootstrap, b_array_bootstrap = TD.get_residuals_and_model_errors_looped(
            dataset, model, X_train, y_train, model_num=trees)

        print("Summary:")
        print("a_bayes = %.3f +/- %.3f" % (np.mean(a_array_bayes), np.std(a_array_bayes)))
        print("b_bayes = %.3f +/- %.3f" % (np.mean(b_array_bayes), np.std(b_array_bayes)))
        print("a_bootstrap = %.3f +/- %.3f" % (np.mean(a_array_bootstrap), np.std(a_array_bootstrap)))
        print("b_bootstrap = %.3f +/- %.3f" % (np.mean(b_array_bootstrap), np.std(b_array_bootstrap)))

        # save data
        save_realdata_both(a_array_bayes, b_array_bayes, Test_residuals, Test_model_errors_unscaled_bayes, Test_model_errors_scaled_bayes,
                      a_array_bootstrap, b_array_bootstrap, Test_model_errors_unscaled_bootstrap, Test_model_errors_scaled_bootstrap)

        # Make scaled and unscaled test data plots
        MP = mp.MakePlot()
        # overlay plots
        MP.make_rstat_overlay(Test_residuals, Test_model_errors_unscaled_bayes, Test_model_errors_scaled_bayes,
                              "{}, {}".format(model, dataset),
                              save=save_plot,
                              file_name='{}/rstat_bayes.png'.format(path))

        MP.make_rve_overlay(Test_residuals, Test_model_errors_unscaled_bayes, Test_model_errors_scaled_bayes,
                            "{}, {}".format(model, dataset),
                            save=save_plot,
                            file_name='{}/RvE_bayes.png'.format(path))

        MP.make_rstat_overlay(Test_residuals, Test_model_errors_unscaled_bootstrap, Test_model_errors_scaled_bootstrap,
                              "{}, {}".format(model, dataset),
                              save=save_plot,
                              file_name='{}/rstat_bayes.png'.format(path))

        MP.make_rve_overlay(Test_residuals, Test_model_errors_unscaled_bootstrap, Test_model_errors_scaled_bootstrap,
                            "{}, {}".format(model, dataset),
                            save=save_plot,
                            file_name='{}/RvE_bayes.png'.format(path))

    else:
        print("Starting run for {} on {} data.".format(model, dataset))
        # Get CV residuals and model errors
        CVD = cvd.CVData()
        CV_residuals, CV_model_errors_bayes, CV_model_errors_bootstrap = CVD.get_residuals_and_model_errors(model, X_train, y_train, model_num=trees)

        # Scale residuals and model errors by data set standard deviation
        stdev = np.std(y_train)
        CV_residuals = CV_residuals / stdev
        CV_model_errors_bayes = CV_model_errors_bayes / stdev
        CV_model_errors_bootstrap = CV_model_errors_bootstrap / stdev

        # Get correction factors
        CF_bayes = cf.CorrectionFactors(CV_residuals, CV_model_errors_bayes)
        CF_bootstrap = cf.CorrectionFactors(CV_residuals, CV_model_errors_bootstrap)
        a_bayes, b_bayes, r_squared_bayes = CF_bayes.nll()
        a_bootstrap, b_bootstrap, r_squared_bootstrap = CF_bootstrap.nll()
        print('Correction Factors:')
        print('a_bayes: ' + str(a_bayes))
        print('b_bayes: ' + str(b_bayes))

        # Get test data residuals and model errors
        TD = td.TestData()
        Test_residuals, Test_model_errors_bayes, Test_model_errors_bootstrap = TD.get_residuals_and_model_errors(model, X_train, y_train, X_test, y_test,
                                                                              model_num=trees)
        # Scale by standard deviation
        Test_residuals = Test_residuals / stdev
        Test_model_errors_bayes = Test_model_errors_bayes / stdev
        Test_model_errors_bootstrap = Test_model_errors_bootstrap / stdev

        # Save data from run
        save_syntheticdata_both(a_bayes, b_bayes, Test_residuals, Test_model_errors_bayes, Test_model_errors_bayes * a_bayes + b_bayes,
                                a_bootstrap, b_bootstrap, Test_model_errors_bootstrap, Test_model_errors_bootstrap * a_bootstrap + b_bootstrap)

        # Make scaled and unscaled test data plots
        MP = mp.MakePlot()
        MP.make_rstat_overlay(Test_residuals, Test_model_errors_bayes, Test_model_errors_bayes * a_bayes + b_bayes,
                              "GPR Friedman -- Single Predictor with Bayesian Errors", save=save_plot,
                              file_name='{}/rstat_bayes.png'.format(path))

        MP.make_rve_overlay(Test_residuals, Test_model_errors_bayes, Test_model_errors_bayes * a_bayes + b_bayes,
                            "GPR Friedman -- Single Predictor with Bayesian Errors",
                            save=save_plot,
                            file_name='{}/RvE_bayes.png'.format(path))

        MP.make_rstat_overlay(Test_residuals, Test_model_errors_bootstrap, Test_model_errors_bootstrap * a_bootstrap + b_bootstrap,
                              "GPR Friedman -- Single Predictor with Bootstrap Errors", save=save_plot,
                              file_name='{}/rstat_bootstrap.png'.format(path))

        MP.make_rve_overlay(Test_residuals, Test_model_errors_bootstrap, Test_model_errors_bootstrap * a_bootstrap + b_bootstrap,
                            "GPR Friedman -- Single Predictor with Bootstrap Errors",
                            save=save_plot,
                            file_name='{}/RvE_bootstrap.png'.format(path))
