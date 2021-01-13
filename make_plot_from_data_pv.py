import matplotlib
from package import MakePlot as mp
import numpy as np

model = "GPR"

unscaled_model_errors = np.load('data_for_paper_plots/Perovskite/{}/Test/Test_model_errors_unscaled.npy'.format(model))
scaled_model_errors = np.load('data_for_paper_plots/Perovskite/{}/Test/Test_model_errors_scaled.npy'.format(model))
residuals = np.load('data_for_paper_plots/Perovskite/{}/Test/Test_residuals.npy'.format(model))
a = np.load('data_for_paper_plots/Perovskite/{}/Test/a.npy'.format(model))
b = np.load('data_for_paper_plots/Perovskite/{}/Test/b.npy'.format(model))

# scaled_model_errors = model_errors * a[0] + b[0]

MP = mp.MakePlot()

MP.make_rve_with_bin_counts_and_slope_1_line(residuals, unscaled_model_errors, "{}, Perovskite, Unscaled".format(model))
MP.make_rve_with_bin_counts_and_slope_1_line(residuals, scaled_model_errors, "{}, Perovskite, Scaled".format(model))
