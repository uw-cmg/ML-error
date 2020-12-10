import numpy as np
from package import MakePlot as mp

datasets = ["Diffusion", "Friedman_500", "Perovskite"]
models = ["LR", "RF"]

for dataset in datasets:
    for model in models:

        a_nll = np.load('data_for_paper_plots/{}/{}/Convergence/a_nll.npy'.format(dataset, model))
        b_nll = np.load('data_for_paper_plots/{}/{}/Convergence/b_nll.npy'.format(dataset, model))

        # Create and save plots
        MP = mp.MakePlot()

        MP.make_convergence_plot(a_nll, "{}, {}, NLL Optimization".format(model, dataset), "a (slope)", save=True,
                                 file_name='Supplemental_Info/{}/5-fold/{}/Convergence_Plots/a_nll'.format(dataset, model))
        MP.make_convergence_plot(b_nll, "{}, {}, NLL Optimization".format(model, dataset), "b (intercept)", save=True,
                                 file_name='Supplemental_Info/{}/5-fold/{}/Convergence_Plots/b_nll'.format(dataset, model))
