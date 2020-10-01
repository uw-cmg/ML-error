import numpy as np
from package import MakePlot as mp

# Load data
#a_direct = np.load('friedman_500_data/a_direct.npy')
#b_direct = np.load('friedman_500_data/b_direct.npy')
#r_squared_direct = np.load('friedman_500_data/r_squared_direct.npy')
#a_direct_unscaled = np.load('friedman_500_data/a_direct_unscaled.npy')
#b_direct_unscaled = np.load('friedman_500_data/b_direct_unscaled.npy')
#r_squared_direct_unscaled = np.load('friedman_500_data/r_squared_direct_unscaled.npy')
#a_rve = np.load('friedman_500_data/a_rve.npy')
#b_rve = np.load('friedman_500_data/b_rve.npy')
#r_squared_rve = np.load('friedman_500_data/r_squared_rve.npy')
a_nll = np.load('friedman_500_data/a_nll.npy')
b_nll = np.load('friedman_500_data/b_nll.npy')


# Create and save plots
MP = mp.MakePlot()
#MP.make_convergence_plot(a_direct, "RF, Direct Optimization", "a (slope)", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/a_direct.png')
#MP.make_convergence_plot(b_direct, "RF, Direct Optimization", "b (intercept)", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/b_direct.png')
#MP.make_convergence_plot(r_squared_direct, "RF, Direct Optimization", "r^2", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/r_squared_direct.png')

#MP.make_convergence_plot(a_direct_unscaled, "RF, Direct Optimization (optimized before std scaling)", "a (slope)", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/a_direct_unscaled.png')
#MP.make_convergence_plot(b_direct_unscaled, "RF, Direct Optimization (optimized before std scaling)", "b (intercept)", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/b_direct_unscaled.png')
#MP.make_convergence_plot(r_squared_direct_unscaled, "RF, Direct Optimization (optimized before std scaling)", "r^2", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/r_squared_direct_unscaled.png')

#MP.make_convergence_plot(a_rve, "RF, Binned RvE Linear Fit", "a (slope)", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/a_rve.png')
#MP.make_convergence_plot(b_rve, "RF, Binned RvE Linear Fit", "b (intercept)", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/b_rve.png')
#MP.make_convergence_plot(r_squared_rve, "RF, Binned RvE Linear Fit", "r^2", save=True, file_name='Supplemental_Info/Friedman_500/5-Fold/RF/Convergence_Plots/r_squared_rve.png')

MP.make_convergence_plot(a_nll, "RF, NLL Optimization", "a (slope)", save=True, file_name='plots/nll_friedman_500/convergence/a_nll')
MP.make_convergence_plot(b_nll, "RF, NLL Optimization", "b (intercept)", save=True, file_name='plots/nll_friedman_500/convergence/b_nll')
