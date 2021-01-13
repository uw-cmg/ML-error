from package import ConvergenceData as cd
from package import MakePlot as mp
import numpy as np


datasets = ["Friedman_500"]
models = ["GPR"]

for dataset in datasets:
    for model in models:
        print("Starting convergence test runs for {}, {}".format(dataset, model))

        # Load data
        if dataset == "Friedman_500":
            X_train = np.load('ML-error/friedman_500_data/training_x_values.npy')
            y_train = np.load('ML-error/friedman_500_data/training_y_values.npy')
        elif dataset == "Diffusion":
            X_train = np.load('ML-error/diffusion_data/all_x_values.npy')
            y_train = np.load('ML-error/diffusion_data/all_y_values.npy')
        elif dataset == "Perovskite":
            X_train = np.load('ML-error/perovskite_data/all_x_values.npy')
            y_train = np.load('ML-error/perovskite_data/all_y_values.npy')
        else:
            print("No valid dataset provided. '{}' is not an option for dataset choice.".format(dataset))
            break

        CD = cd.ConvergenceData()

        a_nll, b_nll = CD.nll([50,100,200], model, X_train, y_train, num_averaged=5)

        np.save('ML-error/data_for_paper_plots/{}/{}/Convergence/a_nll.npy'.format(dataset, model), np.asarray(a_nll))
        np.save('ML-error/data_for_paper_plots/{}/{}/Convergence/b_nll.npy'.format(dataset, model), np.asarray(b_nll))

        # Create and save plots
        MP = mp.MakePlot()

        MP.make_convergence_plot(a_nll, "{}, {}, NLL Optimization".format(model, dataset), "a (slope)", save=True,
                                 file_name='ML-error/Supplemental_Info/{}/5-fold/{}/Convergence_Plots/a_nll'.format(dataset, model))
        MP.make_convergence_plot(b_nll, "{}, {}, NLL Optimization", "b (intercept)".format(model, dataset), save=True,
                                 file_name='ML-error/Supplemental_Info/{}/5-fold/{}/Convergence_Plots/b_nll'.format(dataset, model))
