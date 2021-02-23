import numpy as np

models = ["GPR_Bayesian", "RF", "LR", "GPR"]
dataset = "Diffusion"
save_plot = True

for model in models:
    unscaled_model_errors = np.load('data_for_paper_plots/{}/{}/Test/Test_model_errors_unscaled.npy'.format(dataset, model))
    scaled_model_errors = np.load('data_for_paper_plots/{}/{}/Test/Test_model_errors_scaled.npy'.format(dataset, model))
    residuals = np.load('data_for_paper_plots/{}/{}/Test/Test_residuals.npy'.format(dataset, model))
    a = np.load('data_for_paper_plots/{}/{}/Test/a.npy'.format(dataset, model))
    b = np.load('data_for_paper_plots/{}/{}/Test/b.npy'.format(dataset, model))

    combined = []
    for i in range(0, len(residuals)):
        curr = np.asarray([residuals[i], unscaled_model_errors[i], scaled_model_errors[i]])
        combined.append(curr)

    combined = np.asarray(combined)

    combined_calibration = []
    for i in range(0, len(a)):
        curr = np.asarray([i+1, a[i], b[i]])
        combined_calibration.append(curr)

    combined_calibration = np.asarray(combined_calibration)


    np.savetxt("SI/{}/{}/Test/residuals_and_uncertainty_estimates.csv".format(dataset, model), combined,
               header="residual, uncalibrated_uncertainty_estimate, calibrated_uncertainty_estimate",
               delimiter=",")

    np.savetxt("SI/{}/{}/Test/calibration_values.csv".format(dataset, model), combined_calibration, header="cv_iteration, a, b",
               delimiter=",")