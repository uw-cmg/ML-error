import numpy as np
from package import CVData as cvd
from package import CorrectionFactors as cf
from package import MakePlot as mp
from package import TestData as td

# Specify what models to run
# Options: "RF", "GPR", "BT", "LR"
models = ["RF"]
# Specify how much noise to add
noise_scales = [0.2, 0.3, 0.4]

for model in models:
    for noise_scale in noise_scales:
        print("STARTING {} Friedman 500 with {} noise".format(model, noise_scale))
        # Path to save files
        path = 'Supplemental_Info/Friedman_500/5-Fold/{}_{}_noise'.format(model, noise_scale)
        #path = 'plots/'

        # Load data
        X_train = np.load('friedman_500_data/training_x_values.npy')
        y_train = np.load('friedman_500_data/training_y_values_{}_noise.npy'.format(noise_scale))

        # Get CV residuals and model errors
        CVD = cvd.CVData()
        CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors(model, X_train, y_train)

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
        print('r^2: ' + str(r_squared))

        # Scale residuals, errors, and b by data set standard deviation
        #stdev = np.std(y_train)
        #CV_residuals = CV_residuals / stdev
        #CV_model_errors = CV_model_errors / stdev
        #b = b / stdev

        # Save np arrays of unscaled and scaled CV data
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/CV/a'.format(model, noise_scale), np.asarray([a]))
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/CV/b'.format(model, noise_scale), np.asarray([b]))
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/CV/CV_residuals'.format(model, noise_scale), CV_residuals)
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/CV/CV_model_errors'.format(model, noise_scale), CV_model_errors)


        # Make scaled and unscaled CV plots
        MP = mp.MakePlot()
        #unscaled
        MP.make_rve(CV_residuals, CV_model_errors, "{}, {} noise, Friedman 500, Unscaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/unscaled_RvE.png')
        MP.make_rve_bin_counts(CV_model_errors, "{}, {} noise, Friedman 500, Unscaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/unscaled_RvE_bin_counts.png')
        MP.make_rve_with_bin_counts_and_slope_1_line(CV_residuals, CV_model_errors,
                                    "{}, {} noise, Friedman 500, Unscaled".format(model, noise_scale), save=True,
                                    file_name=path + '/CV_Plots/unscaled_RvE_with_counts.png')
        MP.make_rstat(CV_residuals, CV_model_errors, "{}, {} noise, Friedman 500, Unscaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/unscaled_rstat.png')
        #scaled
        MP.make_rve(CV_residuals, CV_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/scaled_RvE.png')
        MP.make_rve_bin_counts(CV_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/scaled_RvE_bin_counts.png')
        MP.make_rve_with_bin_counts_and_slope_1_line(CV_residuals, CV_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/scaled_RvE_with_counts.png')
        MP.make_rstat(CV_residuals, CV_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled".format(model, noise_scale), save=True,
                    file_name=path + '/CV_Plots/scaled_rstat.png')
        #overlay
        MP.make_rstat_overlay(CV_residuals, CV_model_errors, CV_model_errors * a + b,
                      "{}, {} noise, Friedman 500".format(model, noise_scale), save=True,
                      file_name=path + '/CV_Plots/rstat_overlay.png')
        MP.make_rve_overlay(CV_residuals, CV_model_errors, CV_model_errors * a + b,
                                                     "{}, {} noise, Friedman 500".format(model, noise_scale),
                                                     save=True,
                                                     file_name=path + '/CV_Plots/RvE_overlay.png')


        # Load test data
        X_test = np.load('friedman_500_data/test_x_values_hypercube.npy'.format(noise_scale))
        y_test = np.load('friedman_500_data/test_y_values_{}_noise.npy'.format(noise_scale))

        # Get test data residuals and model errors
        TD = td.TestData()
        Test_residuals, Test_model_errors = TD.get_residuals_and_model_errors(model, X_train, y_train, X_test, y_test)

        # Scale by standard deviation
        Test_residuals = Test_residuals / stdev
        Test_model_errors = Test_model_errors / stdev

        # Save np arrays of unscaled and scaled Test data
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/Test/a'.format(model, noise_scale), np.asarray([a]))
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/Test/b'.format(model, noise_scale), np.asarray([b]))
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/Test/Test_residuals'.format(model, noise_scale), Test_residuals)
        np.save('data_for_paper_plots/Friedman_500/{}_{}_noise/Test/Test_model_errors'.format(model, noise_scale), Test_model_errors)

        # Make scaled and unscaled test data plots
        MP.make_rve(Test_residuals, Test_model_errors, "{}, {} noise, Friedman 500, Unscaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/unscaled_RvE.png')
        MP.make_rve_bin_counts(Test_model_errors, "{}, {} noise, Friedman 500, Unscaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/unscaled_RvE_bin_counts.png')
        MP.make_rve_with_bin_counts_and_slope_1_line(Test_residuals, Test_model_errors, "{}, {} noise, Friedman 500, Unscaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/unscaled_RvE_with_counts.png')
        MP.make_rstat(Test_residuals, Test_model_errors, "{}, {} noise, Friedman 500, Unscaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/unscaled_rstat.png')
        #scaled
        MP.make_rve(Test_residuals, Test_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/scaled_RvE.png')
        MP.make_rve_bin_counts(Test_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/scaled_RvE_bin_counts.png')
        MP.make_rve_with_bin_counts_and_slope_1_line(Test_residuals, Test_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/scaled_RvE_with_counts.png')
        MP.make_rstat(Test_residuals, Test_model_errors*a + b, "{}, {} noise, Friedman 500, Scaled, Test Set".format(model, noise_scale), save=True,
                    file_name=path + '/Test_Plots/scaled_rstat.png')
        #overlay
        MP.make_rstat_overlay(Test_residuals, Test_model_errors, Test_model_errors * a + b,
                      "{}, {} noise, Friedman 500, Test Set".format(model, noise_scale), save=True,
                      file_name=path + '/Test_Plots/rstat_overlay.png')
        MP.make_rve_overlay(Test_residuals, Test_model_errors, Test_model_errors * a + b,
                                                     "{}, {} noise, Friedman 500, Test Set".format(model,
                                                                                                           noise_scale),
                                                     save=True,
                                                     file_name=path + '/Test_Plots/RvE_overlay.png')

