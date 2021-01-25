from package import ConvergenceData as cd
from package import MakePlot as mp
import numpy as np


datasets = ["Perovskite"]
models = ["GPR"]

for dataset in datasets:
    for model in models:
        print("Starting convergence test runs for {}, {}".format(dataset, model))

        # Load data
        if dataset == "Friedman_500":
            X_train = np.load('ML-error/friedman_500_data/training_x_values.npy')
            y_train = np.load('ML-error/friedman_500_data/training_y_values.npy')
        elif dataset == "ML-error/Diffusion":
            X_train = np.load('ML-error/diffusion_data/all_x_values.npy')
            y_train = np.load('ML-error/diffusion_data/all_y_values.npy')
        elif dataset == "Perovskite":
            X_train = np.load('ML-error/perovskite_data/all_x_values.npy')
            y_train = np.load('ML-error/perovskite_data/all_y_values.npy')
        else:
            print("No valid dataset provided. '{}' is not an option for dataset choice.".format(dataset))
            break

        CD = cd.ConvergenceData()

