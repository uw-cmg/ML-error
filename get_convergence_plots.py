import numpy as np
from package import MakePlot as mp

# specify datasets to run -- choices = ["Diffusion", "Friedman_500", "Perovskite"]
datasets = ["Friedman_500", "Diffusion", "Perovskite"]
# specify models to run -- choices = ["RF", "LR", "GPR"]
models = ["RF", "LR"]
saveplot = True

for dataset in datasets:
    for model in models:

        a_nll = np.load('data_for_paper_plots/{}/{}/Convergence/a_nll.npy'.format(dataset, model))
        b_nll = np.load('data_for_paper_plots/{}/{}/Convergence/b_nll.npy'.format(dataset, model))

        # Create and save plots
        MP = mp.MakePlot()

        MP.make_convergence_plot(a_nll, "{}, {}, NLL Optimization".format(model, dataset), "a (slope)", save=saveplot,
                                 file_name='Supplemental_Info/{}/5-fold/{}/Convergence_Plots/a_nll'.format(dataset, model))
        MP.make_convergence_plot(b_nll, "{}, {}, NLL Optimization".format(model, dataset), "b (intercept)", save=saveplot,
                                 file_name='Supplemental_Info/{}/5-fold/{}/Convergence_Plots/b_nll'.format(dataset, model))
