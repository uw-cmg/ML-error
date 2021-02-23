import numpy as np


models = ["0.1", "0.5", "1.0", "2.0"]

for model in models:
    unscaled_model_errors = np.load('data_for_paper_plots/Friedman_500/RF_{}_noise/Test/Test_model_errors.npy'.format(model))
    residuals = np.load('data_for_paper_plots/Friedman_500/RF_{}_noise/Test/Test_residuals.npy'.format(model))
    a = np.load('data_for_paper_plots/Friedman_500/RF_{}_noise/Test/a.npy'.format(model))
    b = np.load('data_for_paper_plots/Friedman_500/RF_{}_noise/Test/b.npy'.format(model))

    scaled_model_errors = unscaled_model_errors * a[0] + b[0]

    combined = []
    for i in range(0, len(residuals)):
        curr = np.asarray([residuals[i], unscaled_model_errors[i], scaled_model_errors[i]])
        combined.append(curr)

    combined = np.asarray(combined)

    calibration = np.asarray([np.asarray([a[0], b[0]])])

    np.savetxt("SI/Noisy_Friedman_500/{}_sigma/RF/residuals_and_uncertainty_estimates.csv".format(model), combined, header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("SI/Noisy_Friedman_500/{}_sigma/RF/calibration_values.csv".format(model), calibration, header="a, b", delimiter=",")
