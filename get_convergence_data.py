from package import ConvergenceData as cd
from package import MakePlot as mp
import numpy as np

# specify datasets to run -- choices = ["Diffusion", "Friedman_500", "Perovskite"]
datasets = ["Diffusion", "Friedman_500", "Perovskite"]
# specify models to run -- choices = ["RF", "LR", "GPR"]
models = ["LR", "RF"]

for dataset in datasets:
    for model in models:
        print("Starting convergence test runs for {}, {}".format(dataset, model))

        # Load data
        if dataset == "Friedman_500":
            X_train = np.load('friedman_500_data/training_x_values.npy')
            y_train = np.load('friedman_500_data/training_y_values.npy')
        elif dataset == "Diffusion":
            X_train = np.load('diffusion_data/all_x_values.npy')
            y_train = np.load('diffusion_data/all_y_values.npy')
        elif dataset == "Perovskite":
            X_train = np.load('perovskite_data/all_x_values.npy')
            y_train = np.load('perovskite_data/all_y_values.npy')
        else:
            print("No valid dataset provided. '{}' is not an option for dataset choice.".format(dataset))
            break

        CD = cd.ConvergenceData()

        a_nll, b_nll = CD.nll([50,100,200,500,1000], model, X_train, y_train, num_averaged=10)

        np.save('data_for_paper_plots/{}/{}/Convergence/a_nll.npy'.format(dataset, model), np.asarray(a_nll))
        np.save('data_for_paper_plots/{}/{}/Convergence/b_nll.npy'.format(dataset, model), np.asarray(b_nll))