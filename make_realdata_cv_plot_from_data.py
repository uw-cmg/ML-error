import matplotlib
from package import MakePlot as mp
import numpy as np

models = ["GPR_Bayesian"]
dataset = "Perovskite"
save_plot = True

for model in models:
    unscaled_model_errors = np.load('data_for_paper_plots/{}/{}/CV/CV_model_errors.npy'.format(dataset, model))
    residuals = np.load('data_for_paper_plots/{}/{}/CV/CV_residuals.npy'.format(dataset, model))
    a = np.load('data_for_paper_plots/{}/{}/CV/a.npy'.format(dataset, model))
    b = np.load('data_for_paper_plots/{}/{}/CV/b.npy'.format(dataset, model))

    scaled_model_errors = unscaled_model_errors * a[0] + b[0]

    MP = mp.MakePlot()

    # old plots:
    #MP.make_rstat(residuals, unscaled_model_errors, "{}, {}, Unscaled, Single CV Split".format(model, dataset), save=save_plot, file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/unscaled_rstat.png'.format(dataset, model))
    #MP.make_rstat(residuals, scaled_model_errors, "{}, {}, Scaled, Single CV Split".format(model, dataset), save=save_plot, file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/scaled_rstat.png'.format(dataset, model))
    #MP.make_rve_with_bin_counts_and_slope_1_line(residuals, unscaled_model_errors, "{}, {}, Unscaled, Single CV Split".format(model, dataset), save=save_plot, file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/unscaled_RvE_with_counts.png'.format(dataset, model))
    #MP.make_rve_with_bin_counts_and_slope_1_line(residuals, scaled_model_errors, "{}, {}, Scaled, Single CV Split".format(model, dataset), save=save_plot, file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/scaled_RvE_with_counts.png'.format(dataset, model))

    # overlay plots:
    MP.make_rstat_overlay(residuals, unscaled_model_errors, scaled_model_errors, "{}, {}, Single CV Split".format(model, dataset),
                          save=save_plot,
                          file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/rstat_overlay.png'.format(
                              dataset, model))

    MP.make_rve_overlay(residuals, unscaled_model_errors, scaled_model_errors, "{}, {}, Single CV Split".format(model, dataset),
                        save=save_plot,
                        file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/RvE_overlay.png'.format(
                            dataset, model))
    MP.make_rstat_overlay_with_table(residuals, unscaled_model_errors, scaled_model_errors,
                          "{}, {}, Single CV Split".format(model, dataset),
                          save=save_plot,
                          file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/rstat_overlay_table.png'.format(
                              dataset, model))

    MP.make_rve_overlay_with_table(residuals, unscaled_model_errors, scaled_model_errors,
                        "{}, {}, Single CV Split".format(model, dataset),
                        save=save_plot,
                        file_name='Supplemental_Info/{}/5-Fold/{}/CV_Plots/RvE_overlay_table.png'.format(
                            dataset, model))
