import numpy as np
from package import CVData as cvd
from package import CorrectionFactors as cf
from package import MakePlot as mp
from package import TestData as td

# Specify what models to run
# Options: "RF", "GPR", "BT", "LR"
models = ["RF", "LR"]

for model in models:
    print("STARTING {} Friedman 500".format(model))
    # Path to save files
    path = 'Supplemental_Info/Friedman_500/5-Fold/{}'.format(model)
    #path = 'plots/'

    # Load data
    X_train = np.load('friedman_500_data/training_x_values.npy')
    y_train = np.load('friedman_500_data/training_y_values.npy')

    # Get CV residuals and model errors
    CVD = cvd.CVData()
    CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors(model, X_train, y_train, model_num=200)

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
    np.save('data_for_paper_plots/Friedman_500/{}/CV/a'.format(model), np.asarray([a]))
    np.save('data_for_paper_plots/Friedman_500/{}/CV/b'.format(model), np.asarray([b]))
    np.save('data_for_paper_plots/Friedman_500/{}/CV/CV_residuals'.format(model), CV_residuals)
    np.save('data_for_paper_plots/Friedman_500/{}/CV/CV_model_errors'.format(model), CV_model_errors)


    # Make scaled and unscaled CV plots
    MP = mp.MakePlot()
    #unscaled
    MP.make_rve(CV_residuals, CV_model_errors, "{}, Friedman 500, Unscaled".format(model), save=True,
                file_name=path + '/CV_Plots/unscaled_RvE.png')
    MP.make_rve_bin_counts(CV_model_errors, "{}, Friedman 500, Unscaled".format(model), save=True,
                file_name=path + '/CV_Plots/unscaled_RvE_bin_counts.png')
    MP.make_rve_with_bin_counts(CV_residuals, CV_model_errors, "{}, Friedman 500, Unscaled".format(model), save=True,
                file_name=path + '/CV_Plots/unscaled_RvE_with_counts.png')
    MP.make_rstat(CV_residuals, CV_model_errors, "{}, Friedman 500, Unscaled".format(model), save=True,
                file_name=path + '/CV_Plots/unscaled_rstat.png')
    #scaled
    MP.make_rve(CV_residuals, CV_model_errors*a + b, "{}, Friedman 500, Scaled".format(model), save=True,
                file_name=path + '/CV_Plots/scaled_RvE.png')
    MP.make_rve_bin_counts(CV_model_errors*a + b, "{}, Friedman 500, Scaled".format(model), save=True,
                file_name=path + '/CV_Plots/scaled_RvE_bin_counts.png')
    MP.make_rve_with_bin_counts(CV_residuals, CV_model_errors*a + b, "{}, Friedman 500, Scaled".format(model), save=True,
                file_name=path + '/CV_Plots/scaled_RvE_with_counts.png')
    MP.make_rstat(CV_residuals, CV_model_errors*a + b, "{}, Friedman 500, Scaled".format(model), save=True,
                file_name=path + '/CV_Plots/scaled_rstat.png')

    # Load test data
    X_test = np.load('friedman_500_data/test_x_values_hypercube.npy')
    y_test = np.load('friedman_500_data/test_y_values_hypercube.npy')

    # Get test data residuals and model errors
    TD = td.TestData()
    Test_residuals, Test_model_errors = TD.get_residuals_and_model_errors(model, X_train, y_train, X_test, y_test, model_num=200)

    # Scale by standard deviation
    Test_residuals = Test_residuals / stdev
    Test_model_errors = Test_model_errors / stdev

    # Save np arrays of unscaled and scaled Test data
    np.save('data_for_paper_plots/Friedman_500/{}/Test/a'.format(model), np.asarray([a]))
    np.save('data_for_paper_plots/Friedman_500/{}/Test/b'.format(model), np.asarray([b]))
    np.save('data_for_paper_plots/Friedman_500/{}/Test/Test_residuals'.format(model), Test_residuals)
    np.save('data_for_paper_plots/Friedman_500/{}/Test/Test_model_errors'.format(model), Test_model_errors)

    # Make scaled and unscaled test data plots
    MP.make_rve(Test_residuals, Test_model_errors, "{}, Friedman 500, Unscaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/unscaled_RvE.png')
    MP.make_rve_bin_counts(Test_model_errors, "{}, Friedman 500, Unscaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/unscaled_RvE_bin_counts.png')
    MP.make_rve_with_bin_counts(Test_residuals, Test_model_errors, "{}, Friedman 500, Unscaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/unscaled_RvE_with_counts.png')
    MP.make_rstat(Test_residuals, Test_model_errors, "{}, Friedman 500, Unscaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/unscaled_rstat.png')
    #scaled
    MP.make_rve(Test_residuals, Test_model_errors*a + b, "{}, Friedman 500, Scaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/scaled_RvE.png')
    MP.make_rve_bin_counts(Test_model_errors*a + b, "{}, Friedman 500, Scaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/scaled_RvE_bin_counts.png')
    MP.make_rve_with_bin_counts(Test_residuals, Test_model_errors*a + b, "{}, Friedman 500, Scaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/scaled_RvE_with_counts.png')
    MP.make_rstat(Test_residuals, Test_model_errors*a + b, "{}, Friedman 500, Scaled, Test Set".format(model), save=True,
                file_name=path + '/Test_Plots/scaled_rstat.png')

