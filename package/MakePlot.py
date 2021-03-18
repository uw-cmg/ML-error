from package import rf
import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats

class MakePlot:

    def __init__(self):
        pass

    def make_rve(self, residuals, model_errors, title, save=False, file_name=None, number_of_bins=15):
        # Define input data -- divide by standard deviation
        abs_res = abs(residuals)

        # check to see if number of bins should increase, and increase it if so
        model_errors_sorted = np.sort(model_errors)
        ninety_percentile = int(len(model_errors_sorted) * 0.9)
        ninety_percentile_range = model_errors_sorted[ninety_percentile] - np.amin(model_errors)
        total_range = np.amax(model_errors) - np.amin(model_errors)
        number_of_bins = number_of_bins
        if ninety_percentile_range / total_range < 5 / number_of_bins:
            number_of_bins = int(5 * total_range / ninety_percentile_range)

        # Set bins for calculating RMS
        upperbound = np.amax(model_errors)
        lowerbound = np.amin(model_errors)
        bins = np.linspace(lowerbound, upperbound, number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        digitized = np.digitize(model_errors, bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        bins_present = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                bins_present.append(i)

        # Create array of weights based on counts in each bin
        weights = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                weights.append(np.count_nonzero(digitized == i))

        # Calculate RMS of the absolute residuals
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in
                       range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        # Fit a line to the data
        model = LinearRegression(fit_intercept=True)
        model.fit(binned_model_errors[:, np.newaxis],
                  RMS_abs_res,
                  sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        xfit = binned_model_errors
        yfit = model.predict(xfit[:, np.newaxis])

        # Calculate r^2 value
        r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
        # Calculate slope
        slope = model.coef_
        # Calculate y-intercept
        intercept = model.intercept_

        # Create RvE plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Binned RvE Plot -- {}'.format(title))
        ax.set_xlabel('model error estimates / standard deviation')
        ax.set_ylabel('residuals / standard deviation')
        ax.plot(binned_model_errors, RMS_abs_res, 'o', color='blue')
        ax.plot(xfit, yfit)
        ax.text(0.04, 0.9, 'r^2 = %f' % (r_squared), transform=ax.transAxes)
        ax.text(0.04, 0.85, 'slope = %f' % (slope), transform=ax.transAxes)
        ax.text(0.04, 0.8, 'y-intercept = %f' % (intercept), transform=ax.transAxes)
        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return model_errors, abs_res, r_squared, slope, intercept, binned_model_errors, RMS_abs_res, xfit, yfit

    def make_rve_bin_counts(self, model_errors, title, save=False, file_name=None, number_of_bins=15):
        # check to see if number of bins should increase, and increase it if so
        model_errors_sorted = np.sort(model_errors)
        ninety_percentile = int(len(model_errors_sorted) * 0.9)
        ninety_percentile_range = model_errors_sorted[ninety_percentile] - np.amin(model_errors)
        total_range = np.amax(model_errors) - np.amin(model_errors)
        number_of_bins = number_of_bins
        if ninety_percentile_range / total_range < 5 / number_of_bins:
            number_of_bins = int(5 * total_range / ninety_percentile_range)

        # Create bin counts plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Bin Counts from RvE Plot -- {}'.format(title))
        ax.set_xlabel('model error estimates / standard deviation')
        ax.set_ylabel('bin counts')
        ax.hist(model_errors, bins=number_of_bins, color='blue', edgecolor='black')

        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return 0

    def make_rve_with_bin_counts(self, residuals, model_errors, title, save=False, file_name=None, number_of_bins=15):
        # Define input data -- divide by standard deviation
        abs_res = abs(residuals)

        # check to see if number of bins should increase, and increase it if so
        model_errors_sorted = np.sort(model_errors)
        ninety_percentile = int(len(model_errors_sorted) * 0.9)
        ninety_percentile_range = model_errors_sorted[ninety_percentile] - np.amin(model_errors)
        total_range = np.amax(model_errors) - np.amin(model_errors)
        number_of_bins = number_of_bins
        if ninety_percentile_range / total_range < 5 / number_of_bins:
            number_of_bins = int(5 * total_range / ninety_percentile_range)

        # Set bins for calculating RMS
        upperbound = np.amax(model_errors)
        lowerbound = np.amin(model_errors)
        bins = np.linspace(lowerbound, upperbound, number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        digitized = np.digitize(model_errors, bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        bins_present = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                bins_present.append(i)

        # Create array of weights based on counts in each bin
        weights = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                weights.append(np.count_nonzero(digitized == i))

        # Calculate RMS of the absolute residuals
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in
                       range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        # Fit a line to the data
        model = LinearRegression(fit_intercept=True)
        model.fit(binned_model_errors[:, np.newaxis],
                  RMS_abs_res,
                  sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        xfit = binned_model_errors
        yfit = model.predict(xfit[:, np.newaxis])

        # Calculate r^2 value
        r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
        # Calculate slope
        slope = model.coef_
        # Calculate y-intercept
        intercept = model.intercept_

        # Create RvE plot
        fig = plt.figure()
        ax = fig.add_subplot(3,1,(2,3))
        #ax.set_title('Binned RvE Plot -- {}'.format(title))
        ax.set_xlabel('model error estimates / standard deviation')
        ax.set_ylabel('residuals / standard deviation')
        ax.plot(binned_model_errors, RMS_abs_res, 'o', color='blue')
        ax.plot(xfit, yfit)
        ax.text(0.04, 0.92, 'r^2 = %f' % (r_squared), transform=ax.transAxes)
        ax.text(0.04, 0.85, 'slope = %f' % (slope), transform=ax.transAxes)
        ax.text(0.04, 0.78, 'y-intercept = %f' % (intercept), transform=ax.transAxes)

        ax = fig.add_subplot(3,1,1)
        ax.set_title('Binned RvE Plot -- {}'.format(title))
        #ax.set_xlabel('model error estimates / standard deviation')
        ax.set_xticks([])
        #ax_set_xticklabels([])
        ax.set_ylabel('bin counts')
        ax.hist(model_errors, bins=number_of_bins, color='blue', edgecolor='black')

        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return model_errors, abs_res, r_squared, slope, intercept, binned_model_errors, RMS_abs_res, xfit, yfit

    def make_rstat(self, residuals, model_errors, title, save=False, file_name=None):
        # Eliminate model errors with value 0, so that the ratios can be calculated
        zero_indices = []
        for i in range(0, len(model_errors)):
            if model_errors[i] == 0:
                zero_indices.append(i)
        residuals = np.delete(residuals, zero_indices)
        model_errors = np.delete(model_errors, zero_indices)
        print("{} values deleted before making r-stat plot because model errors were zero.".format(len(zero_indices)))
        # make data for gaussian plot
        gaussian_x = np.linspace(-5, 5, 1000)
        # create plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('r-statistic distribution -- {}'.format(title))
        ax.set_xlabel('residuals / model error estimates')
        ax.set_ylabel('relative counts')
        ax.hist(residuals/model_errors, bins=30, color='blue', edgecolor='black', density=True)
        ax.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='orange')
        ax.text(0.05, 0.9, 'mean = %.3f' % (np.mean(residuals / model_errors)), transform=ax.transAxes)
        ax.text(0.05, 0.85, 'std = %.3f' % (np.std(residuals / model_errors)), transform=ax.transAxes)
        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return 0

    def make_convergence_plot(self, convergence_data, title, parameter, save=False, file_name=None):
        # organize data
        x, y, err = [], [], []
        for i in range(0, len(convergence_data)):
            k = convergence_data[i]
            x.append(k[0])
            y.append(k[1])
            err.append(k[2])
        # make plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('convergence results -- {}'.format(title))
        ax.set_xlabel('number of models')
        ax.set_ylabel(parameter)
        ax.errorbar(x, y, yerr=err, fmt='o')
        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)

    def make_rve_with_bin_counts_and_slope_1_line(self, residuals, model_errors, title, save=False, file_name=None, number_of_bins=15):
        # Define input data -- divide by standard deviation
        abs_res = abs(residuals)

        # check to see if number of bins should increase, and increase it if so
        model_errors_sorted = np.sort(model_errors)
        ninety_percentile = int(len(model_errors_sorted) * 0.9)
        ninety_percentile_range = model_errors_sorted[ninety_percentile] - np.amin(model_errors)
        total_range = np.amax(model_errors) - np.amin(model_errors)
        number_of_bins = number_of_bins
        if ninety_percentile_range / total_range < 5 / number_of_bins:
            number_of_bins = int(5 * total_range / ninety_percentile_range)

        # Set bins for calculating RMS
        upperbound = np.amax(model_errors)
        lowerbound = np.amin(model_errors)
        bins = np.linspace(lowerbound, upperbound, number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        digitized = np.digitize(model_errors, bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        bins_present = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                bins_present.append(i)

        # Create array of weights based on counts in each bin
        weights = []
        for i in range(1, number_of_bins + 1):
            if i in digitized:
                weights.append(np.count_nonzero(digitized == i))

        # Calculate RMS of the absolute residuals
        RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in
                       range(0, len(bins_present))]

        # Set the x-values to the midpoint of each bin
        bin_width = bins[1] - bins[0]
        binned_model_errors = np.zeros(len(bins_present))
        for i in range(0, len(bins_present)):
            curr_bin = bins_present[i]
            binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

        # Fit a line to the data
        model = LinearRegression(fit_intercept=True)
        model.fit(binned_model_errors[:, np.newaxis],
                  RMS_abs_res,
                  sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        xfit = binned_model_errors
        yfit = model.predict(xfit[:, np.newaxis])

        # Calculate r^2 value
        r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
        # Calculate slope
        slope = model.coef_
        # Calculate y-intercept
        intercept = model.intercept_

        # Create RvE plot
        fig = plt.figure()
        ax = fig.add_subplot(3,1,(2,3))
        x = np.linspace(lowerbound, upperbound, 100)
        ax.plot(x, x, color='red', label='identity function')
        #ax.set_title('Binned RvE Plot -- {}'.format(title))
        ax.set_xlabel('model error estimates / standard deviation')
        ax.set_ylabel('RMS residuals / standard deviation')
        ax.plot(binned_model_errors, RMS_abs_res, 'o', color='blue')
        ax.plot(xfit, yfit, color='blue', label='fitted line')
        ax.text(0.04, 0.92, '$R^2$ = %.3f' % (r_squared), transform=ax.transAxes)
        ax.text(0.04, 0.85, 'slope = %.3f' % (slope), transform=ax.transAxes)
        ax.text(0.04, 0.78, 'y-intercept = %.3f' % (intercept), transform=ax.transAxes)
        ax.legend(loc="lower right")

        ax = fig.add_subplot(3,1,1)
        ax.set_title('Binned RvE Plot -- {}'.format(title))
        #ax.set_xlabel('model error estimates / standard deviation')
        ax.set_xticks([])
        #ax_set_xticklabels([])
        ax.set_ylabel('bin counts')
        ax.hist(model_errors, bins=number_of_bins, color='blue', edgecolor='black')

        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return model_errors, abs_res, r_squared, slope, intercept, binned_model_errors, RMS_abs_res, xfit, yfit

    def make_rstat_overlay(self, residuals, unscaled_model_errors, scaled_model_errors, title, save=False, file_name=None):
        # Eliminate model errors with value 0, so that the ratios can be calculated
        zero_indices = []
        for i in range(0, len(unscaled_model_errors)):
            if unscaled_model_errors[i] == 0:
                zero_indices.append(i)
        unscaled_residuals = np.delete(residuals, zero_indices)
        unscaled_model_errors = np.delete(unscaled_model_errors, zero_indices)
        print("{} values deleted before making r-stat plot because unscaled model errors were zero.".format(len(zero_indices)))
        scaled_zero_indices = []
        for i in range(0, len(scaled_model_errors)):
            if scaled_model_errors[i] == 0:
                scaled_zero_indices.append(i)
        scaled_residuals = np.delete(residuals, scaled_zero_indices)
        scaled_model_errors = np.delete(scaled_model_errors, scaled_zero_indices)
        print("{} values deleted before making r-stat plot because scaled model errors were zero.".format(len(zero_indices)))
        # make data for gaussian plot
        gaussian_x = np.linspace(-5, 5, 1000)
        # create plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('r-statistic distribution -- {}'.format(title))
        ax.set_xlabel('residuals / uncertainty estimates')
        ax.set_ylabel('relative counts')
        ax.hist(unscaled_residuals / unscaled_model_errors, bins=30, color='grey', edgecolor='black', density=True, alpha=0.5, label='uncalibrated')
        ax.hist(scaled_residuals / scaled_model_errors, bins=30, color='blue', edgecolor='black', density=True, alpha=0.5, label='calibrated')
        ax.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='standard normal pdf', color='orange')
        ax.legend(loc="upper left")
        #ax.text(0.05, 0.9, 'mean = %.3f' % (np.mean(residuals / model_errors)), transform=ax.transAxes)
        #ax.text(0.05, 0.85, 'std = %.3f' % (np.std(residuals / model_errors)), transform=ax.transAxes)
        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return 0

    def make_rve_overlay(self, residuals, unscaled_model_errors, scaled_model_errors, title, save=False, file_name=None, number_of_bins=15):
        # Define input data
        abs_res = abs(residuals)

        ####################### Unscaled #####################

        # check to see if number of bins should increase, and increase it if so
        unscaled_model_errors_sorted = np.sort(unscaled_model_errors)
        unscaled_ninety_percentile = int(len(unscaled_model_errors_sorted) * 0.9)
        unscaled_ninety_percentile_range = unscaled_model_errors_sorted[unscaled_ninety_percentile] - np.amin(unscaled_model_errors)
        unscaled_total_range = np.amax(unscaled_model_errors) - np.amin(unscaled_model_errors)
        unscaled_number_of_bins = number_of_bins
        if unscaled_ninety_percentile_range / unscaled_total_range < 5 / unscaled_number_of_bins:
            unscaled_number_of_bins = int(5 * unscaled_total_range / unscaled_ninety_percentile_range)

        # Set bins for calculating RMS
        unscaled_upperbound = np.amax(unscaled_model_errors)
        unscaled_lowerbound = np.amin(unscaled_model_errors)
        unscaled_bins = np.linspace(unscaled_lowerbound, unscaled_upperbound, unscaled_number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        unscaled_digitized = np.digitize(unscaled_model_errors, unscaled_bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        unscaled_bins_present = []
        for i in range(1, unscaled_number_of_bins + 1):
            if i in unscaled_digitized:
                unscaled_bins_present.append(i)

        # Create array of weights based on counts in each bin
        unscaled_weights = []
        for i in range(1, unscaled_number_of_bins + 1):
            if i in unscaled_digitized:
                unscaled_weights.append(np.count_nonzero(unscaled_digitized == i))

        # Calculate RMS of the absolute residuals
        unscaled_RMS_abs_res = [np.sqrt((abs_res[unscaled_digitized == unscaled_bins_present[i]] ** 2).mean()) for i in
                       range(0, len(unscaled_bins_present))]

        # Set the x-values to the midpoint of each bin
        unscaled_bin_width = unscaled_bins[1] - unscaled_bins[0]
        unscaled_binned_model_errors = np.zeros(len(unscaled_bins_present))
        for i in range(0, len(unscaled_bins_present)):
            unscaled_curr_bin = unscaled_bins_present[i]
            unscaled_binned_model_errors[i] = unscaled_bins[unscaled_curr_bin - 1] + unscaled_bin_width / 2

        # Fit a line to the data
        unscaled_model = LinearRegression(fit_intercept=True)
        unscaled_model.fit(unscaled_binned_model_errors[:, np.newaxis],
                  unscaled_RMS_abs_res,
                  sample_weight=unscaled_weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        unscaled_xfit = unscaled_binned_model_errors
        unscaled_yfit = unscaled_model.predict(unscaled_xfit[:, np.newaxis])

        # Calculate r^2 value
        unscaled_r_squared = r2_score(unscaled_RMS_abs_res, unscaled_yfit, sample_weight=unscaled_weights)
        # Calculate slope
        unscaled_slope = unscaled_model.coef_
        # Calculate y-intercept
        unscaled_intercept = unscaled_model.intercept_

        ####################### Scaled #####################

        # check to see if number of bins should increase, and increase it if so
        scaled_model_errors_sorted = np.sort(scaled_model_errors)
        scaled_ninety_percentile = int(len(scaled_model_errors_sorted) * 0.9)
        scaled_ninety_percentile_range = scaled_model_errors_sorted[scaled_ninety_percentile] - np.amin(scaled_model_errors)
        scaled_total_range = np.amax(scaled_model_errors) - np.amin(scaled_model_errors)
        scaled_number_of_bins = number_of_bins
        if scaled_ninety_percentile_range / scaled_total_range < 5 / scaled_number_of_bins:
            scaled_number_of_bins = int(5 * scaled_total_range / scaled_ninety_percentile_range)

        # Set bins for calculating RMS
        scaled_upperbound = np.amax(scaled_model_errors)
        scaled_lowerbound = np.amin(scaled_model_errors)
        scaled_bins = np.linspace(scaled_lowerbound, scaled_upperbound, scaled_number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        scaled_digitized = np.digitize(scaled_model_errors, scaled_bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        scaled_bins_present = []
        for i in range(1, scaled_number_of_bins + 1):
            if i in scaled_digitized:
                scaled_bins_present.append(i)

        # Create array of weights based on counts in each bin
        scaled_weights = []
        for i in range(1, scaled_number_of_bins + 1):
            if i in scaled_digitized:
                scaled_weights.append(np.count_nonzero(scaled_digitized == i))

        # Calculate RMS of the absolute residuals
        scaled_RMS_abs_res = [np.sqrt((abs_res[scaled_digitized == scaled_bins_present[i]] ** 2).mean()) for i in
                       range(0, len(scaled_bins_present))]

        # Set the x-values to the midpoint of each bin
        scaled_bin_width = scaled_bins[1] - scaled_bins[0]
        scaled_binned_model_errors = np.zeros(len(scaled_bins_present))
        for i in range(0, len(scaled_bins_present)):
            scaled_curr_bin = scaled_bins_present[i]
            scaled_binned_model_errors[i] = scaled_bins[scaled_curr_bin - 1] + scaled_bin_width / 2

        # Fit a line to the data
        scaled_model = LinearRegression(fit_intercept=True)
        scaled_model.fit(scaled_binned_model_errors[:, np.newaxis],
                  scaled_RMS_abs_res,
                  sample_weight=scaled_weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        scaled_xfit = scaled_binned_model_errors
        scaled_yfit = scaled_model.predict(scaled_xfit[:, np.newaxis])

        # Calculate r^2 value
        scaled_r_squared = r2_score(scaled_RMS_abs_res, scaled_yfit, sample_weight=scaled_weights)
        # Calculate slope
        scaled_slope = scaled_model.coef_
        # Calculate y-intercept
        scaled_intercept = scaled_model.intercept_

        ################# Making plot ##################

        # define overall lower and upper bounds
        lowerbound = min(unscaled_lowerbound, scaled_lowerbound)
        upperbound = max(unscaled_upperbound, scaled_upperbound)

        # Create RvE plot
        fig = plt.figure()
        ax = fig.add_subplot(3,1,(2,3))
        x = np.linspace(lowerbound, upperbound, 100)
        ax.plot(x, x, color='red', label='identity function', alpha=0.5, linestyle='dashed')
        #ax.set_title('Binned RvE Plot -- {}'.format(title))
        ax.set_xlabel('uncertainty estimates')
        ax.set_ylabel('RMS residuals')
        # unscaled stuff
        # separate well-sampled data (> 30 points) from poorly-sampled data
        unscaled_binned_model_errors_under30 = []
        unscaled_binned_model_errors_over30 = []
        unscaled_RMS_abs_res_under30 = []
        unscaled_RMS_abs_res_over30 = []
        for i in range(0, len(unscaled_binned_model_errors)):
            if unscaled_weights[i] > 30:
                unscaled_binned_model_errors_over30.append(unscaled_binned_model_errors[i])
                unscaled_RMS_abs_res_over30.append(unscaled_RMS_abs_res[i])
            else:
                unscaled_binned_model_errors_under30.append(unscaled_binned_model_errors[i])
                unscaled_RMS_abs_res_under30.append(unscaled_RMS_abs_res[i])
        #ax.plot(unscaled_binned_model_errors, unscaled_RMS_abs_res, 'o', color='grey', alpha=0.5)
        ax.plot(unscaled_binned_model_errors_over30, unscaled_RMS_abs_res_over30, 'o', mec='grey', mfc='grey', alpha=0.5)
        ax.plot(unscaled_binned_model_errors_under30, unscaled_RMS_abs_res_under30, 'o', mec='grey', mfc='none', alpha=0.5)
        ax.plot(unscaled_xfit, unscaled_yfit, color='grey', label='fitted line (uncalibrated)', alpha=0.5)
        #ax.text(0.04, 0.92, '$R^2$ = %.3f' % (r_squared), transform=ax.transAxes)
        #ax.text(0.04, 0.85, 'slope = %.3f' % (slope), transform=ax.transAxes)
        #ax.text(0.04, 0.78, 'y-intercept = %.3f' % (intercept), transform=ax.transAxes)
        # scaled stuff
        # separate well-sampled data (> 30 points) from poorly-sampled data
        scaled_binned_model_errors_under30 = []
        scaled_binned_model_errors_over30 = []
        scaled_RMS_abs_res_under30 = []
        scaled_RMS_abs_res_over30 = []
        bin_counts_under_30 = []
        bin_counts_over_30 = []
        for i in range(0, len(scaled_binned_model_errors)):
            if scaled_weights[i] > 30:
                scaled_binned_model_errors_over30.append(scaled_binned_model_errors[i])
                scaled_RMS_abs_res_over30.append(scaled_RMS_abs_res[i])
                bin_counts_over_30.append(scaled_weights[i])
            else:
                scaled_binned_model_errors_under30.append(scaled_binned_model_errors[i])
                scaled_RMS_abs_res_under30.append(scaled_RMS_abs_res[i])
                bin_counts_under_30.append(scaled_weights[i])
        #ax.plot(scaled_binned_model_errors, scaled_RMS_abs_res, 'o', color='blue', alpha=0.5)
        ax.plot(scaled_binned_model_errors_over30, scaled_RMS_abs_res_over30, 'o', mec='blue', mfc='blue', alpha=0.75)
        ax.plot(scaled_binned_model_errors_under30, scaled_RMS_abs_res_under30, 'o', mec='blue', mfc='none', alpha=0.75)
        ax.plot(scaled_xfit, scaled_yfit, color='blue', label='fitted line (calibrated)', alpha=0.5)
        #ax.text(0.04, 0.92, '$R^2$ = %.3f' % (r_squared), transform=ax.transAxes)
        #ax.text(0.04, 0.85, 'slope = %.3f' % (slope), transform=ax.transAxes)
        #ax.text(0.04, 0.78, 'y-intercept = %.3f' % (intercept), transform=ax.transAxes)


        ax.legend(loc="upper left")

        ax = fig.add_subplot(3,1,1)
        ax.set_title('Binned RvE Plot -- {}'.format(title))
        #ax.set_xlabel('model error estimates / standard deviation')
        ax.set_xticks([])
        #ax_set_xticklabels([])
        ax.set_ylabel('bin counts')
        # unscaled
        ax.hist(unscaled_model_errors, bins=unscaled_number_of_bins, color='grey', edgecolor='black', alpha=0.5, label='uncalibrated')
        # scaled
        ax.hist(scaled_model_errors, bins=scaled_number_of_bins, color='blue', edgecolor='black', alpha=0.5, label='calibrated')
        ax.legend(loc="upper right")


        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)

        ########################## Print out stats for lines fit to all and then just well-sampled data ########################

        # Fit a line to just the well-sampled calibrated data
        scaled_model_over30 = LinearRegression(fit_intercept=True)
        scaled_model_over30.fit(np.asarray(scaled_binned_model_errors_over30).reshape(-1,1), np.asarray(scaled_RMS_abs_res_over30), sample_weight=np.asarray(bin_counts_over_30))  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        scaled_xfit_over30 = np.asarray(scaled_binned_model_errors_over30).reshape(-1,1)
        scaled_yfit_over30 = scaled_model_over30.predict(scaled_xfit_over30)

        # Calculate r^2 value
        scaled_r_squared_over30 = r2_score(scaled_RMS_abs_res_over30, scaled_yfit_over30, sample_weight=np.asarray(bin_counts_over_30))
        # Calculate slope
        scaled_slope_over30 = scaled_model_over30.coef_
        # Calculate y-intercept
        scaled_intercept_over30 = scaled_model_over30.intercept_

        print("Calibrated fit lines values:")
        print("Line fit to all points (weighted by bin counts):")
        print("slope = {}".format(scaled_slope))
        print("y-intercept = {}".format(scaled_intercept))
        print("r^2 = {}".format(scaled_r_squared))
        print("Line fit to only well-sampled points (weighted by bin counts):")
        print("slope = {}".format(scaled_slope_over30))
        print("y-intercept = {}".format(scaled_intercept_over30))
        print("r^2 = {}".format(scaled_r_squared_over30))

        return unscaled_model_errors, abs_res, unscaled_r_squared, unscaled_slope, unscaled_intercept, unscaled_binned_model_errors, unscaled_RMS_abs_res, unscaled_xfit, unscaled_yfit, scaled_model_errors, abs_res, scaled_r_squared, scaled_slope, scaled_intercept, scaled_binned_model_errors, scaled_RMS_abs_res, scaled_xfit, scaled_yfit

    def make_rstat_overlay_with_table(self, residuals, unscaled_model_errors, scaled_model_errors, title, save=False, file_name=None):
        # Eliminate model errors with value 0, so that the ratios can be calculated
        zero_indices = []
        for i in range(0, len(unscaled_model_errors)):
            if unscaled_model_errors[i] == 0:
                zero_indices.append(i)
        unscaled_residuals = np.delete(residuals, zero_indices)
        unscaled_model_errors = np.delete(unscaled_model_errors, zero_indices)
        print("{} values deleted before making r-stat plot because unscaled model errors were zero.".format(len(zero_indices)))
        scaled_zero_indices = []
        for i in range(0, len(scaled_model_errors)):
            if scaled_model_errors[i] == 0:
                scaled_zero_indices.append(i)
        scaled_residuals = np.delete(residuals, scaled_zero_indices)
        scaled_model_errors = np.delete(scaled_model_errors, scaled_zero_indices)
        print("{} values deleted before making r-stat plot because scaled model errors were zero.".format(len(zero_indices)))
        # make data for gaussian plot
        gaussian_x = np.linspace(-5, 5, 1000)
        # create plot
        fig = plt.figure()
        ax = fig.add_subplot(5,10,(2,40))
        ax.set_title('r-statistic distribution -- {}'.format(title))
        ax.set_xlabel('residuals / uncertainty estimates')
        ax.set_ylabel('relative counts')
        ax.hist(unscaled_residuals / unscaled_model_errors, bins=30, color='grey', edgecolor='black', density=True, alpha=0.5, label='uncalibrated')
        ax.hist(scaled_residuals / scaled_model_errors, bins=30, color='blue', edgecolor='black', density=True, alpha=0.5, label='calibrated')
        ax.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='standard normal pdf', color='orange')
        ax.legend(loc="upper left")
        #ax.text(0.05, 0.9, 'mean = %.3f' % (np.mean(residuals / model_errors)), transform=ax.transAxes)
        #ax.text(0.05, 0.85, 'std = %.3f' % (np.std(residuals / model_errors)), transform=ax.transAxes)
        # Add table
        colLabels = ['mean', 'standard deviation']
        rowLabels = ['uncalibrated', 'calibrated']
        unscaled = ['%.3f' % (np.mean(unscaled_residuals / unscaled_model_errors)), '%.3f' % (np.std(unscaled_residuals / unscaled_model_errors))]
        scaled = ['%.3f' % (np.mean(scaled_residuals / scaled_model_errors)), '%.3f' % (np.std(scaled_residuals / scaled_model_errors))]
        cellText = [unscaled, scaled]
        ax.table(cellText=cellText, rowLoc='center', rowLabels=rowLabels, colWidths=[.5, .5], colLabels=colLabels, colLoc='center', loc='bottom', bbox=[0, -0.4, 1, 0.2])
        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return 0

    def make_rve_overlay_with_table(self, residuals, unscaled_model_errors, scaled_model_errors, title, save=False, file_name=None, number_of_bins=15):
        # Define input data
        abs_res = abs(residuals)

        ####################### Unscaled #####################

        # check to see if number of bins should increase, and increase it if so
        unscaled_model_errors_sorted = np.sort(unscaled_model_errors)
        unscaled_ninety_percentile = int(len(unscaled_model_errors_sorted) * 0.9)
        unscaled_ninety_percentile_range = unscaled_model_errors_sorted[unscaled_ninety_percentile] - np.amin(unscaled_model_errors)
        unscaled_total_range = np.amax(unscaled_model_errors) - np.amin(unscaled_model_errors)
        unscaled_number_of_bins = number_of_bins
        if unscaled_ninety_percentile_range / unscaled_total_range < 5 / unscaled_number_of_bins:
            unscaled_number_of_bins = int(5 * unscaled_total_range / unscaled_ninety_percentile_range)

        # Set bins for calculating RMS
        unscaled_upperbound = np.amax(unscaled_model_errors)
        unscaled_lowerbound = np.amin(unscaled_model_errors)
        unscaled_bins = np.linspace(unscaled_lowerbound, unscaled_upperbound, unscaled_number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        unscaled_digitized = np.digitize(unscaled_model_errors, unscaled_bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        unscaled_bins_present = []
        for i in range(1, unscaled_number_of_bins + 1):
            if i in unscaled_digitized:
                unscaled_bins_present.append(i)

        # Create array of weights based on counts in each bin
        unscaled_weights = []
        for i in range(1, unscaled_number_of_bins + 1):
            if i in unscaled_digitized:
                unscaled_weights.append(np.count_nonzero(unscaled_digitized == i))

        # Calculate RMS of the absolute residuals
        unscaled_RMS_abs_res = [np.sqrt((abs_res[unscaled_digitized == unscaled_bins_present[i]] ** 2).mean()) for i in
                       range(0, len(unscaled_bins_present))]

        # Set the x-values to the midpoint of each bin
        unscaled_bin_width = unscaled_bins[1] - unscaled_bins[0]
        unscaled_binned_model_errors = np.zeros(len(unscaled_bins_present))
        for i in range(0, len(unscaled_bins_present)):
            unscaled_curr_bin = unscaled_bins_present[i]
            unscaled_binned_model_errors[i] = unscaled_bins[unscaled_curr_bin - 1] + unscaled_bin_width / 2

        # Fit a line to the data
        unscaled_model = LinearRegression(fit_intercept=True)
        unscaled_model.fit(unscaled_binned_model_errors[:, np.newaxis],
                  unscaled_RMS_abs_res,
                  sample_weight=unscaled_weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        unscaled_xfit = unscaled_binned_model_errors
        unscaled_yfit = unscaled_model.predict(unscaled_xfit[:, np.newaxis])

        # Calculate r^2 value
        unscaled_r_squared = r2_score(unscaled_RMS_abs_res, unscaled_yfit, sample_weight=unscaled_weights)
        # Calculate slope
        unscaled_slope = unscaled_model.coef_
        # Calculate y-intercept
        unscaled_intercept = unscaled_model.intercept_

        ####################### Scaled #####################

        # check to see if number of bins should increase, and increase it if so
        scaled_model_errors_sorted = np.sort(scaled_model_errors)
        scaled_ninety_percentile = int(len(scaled_model_errors_sorted) * 0.9)
        scaled_ninety_percentile_range = scaled_model_errors_sorted[scaled_ninety_percentile] - np.amin(scaled_model_errors)
        scaled_total_range = np.amax(scaled_model_errors) - np.amin(scaled_model_errors)
        scaled_number_of_bins = number_of_bins
        if scaled_ninety_percentile_range / scaled_total_range < 5 / scaled_number_of_bins:
            scaled_number_of_bins = int(5 * scaled_total_range / scaled_ninety_percentile_range)

        # Set bins for calculating RMS
        scaled_upperbound = np.amax(scaled_model_errors)
        scaled_lowerbound = np.amin(scaled_model_errors)
        scaled_bins = np.linspace(scaled_lowerbound, scaled_upperbound, scaled_number_of_bins, endpoint=False)

        # Create a vector determining bin of each data point
        scaled_digitized = np.digitize(scaled_model_errors, scaled_bins)

        # Record which bins contain data (to avoid trying to do calculations on empty bins)
        scaled_bins_present = []
        for i in range(1, scaled_number_of_bins + 1):
            if i in scaled_digitized:
                scaled_bins_present.append(i)

        # Create array of weights based on counts in each bin
        scaled_weights = []
        for i in range(1, scaled_number_of_bins + 1):
            if i in scaled_digitized:
                scaled_weights.append(np.count_nonzero(scaled_digitized == i))

        # Calculate RMS of the absolute residuals
        scaled_RMS_abs_res = [np.sqrt((abs_res[scaled_digitized == scaled_bins_present[i]] ** 2).mean()) for i in
                       range(0, len(scaled_bins_present))]

        # Set the x-values to the midpoint of each bin
        scaled_bin_width = scaled_bins[1] - scaled_bins[0]
        scaled_binned_model_errors = np.zeros(len(scaled_bins_present))
        for i in range(0, len(scaled_bins_present)):
            scaled_curr_bin = scaled_bins_present[i]
            scaled_binned_model_errors[i] = scaled_bins[scaled_curr_bin - 1] + scaled_bin_width / 2

        # Fit a line to the data
        scaled_model = LinearRegression(fit_intercept=True)
        scaled_model.fit(scaled_binned_model_errors[:, np.newaxis],
                  scaled_RMS_abs_res,
                  sample_weight=scaled_weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
        scaled_xfit = scaled_binned_model_errors
        scaled_yfit = scaled_model.predict(scaled_xfit[:, np.newaxis])

        # Calculate r^2 value
        scaled_r_squared = r2_score(scaled_RMS_abs_res, scaled_yfit, sample_weight=scaled_weights)
        # Calculate slope
        scaled_slope = scaled_model.coef_
        # Calculate y-intercept
        scaled_intercept = scaled_model.intercept_

        ################# Making plot ##################

        # define overall lower and upper bounds
        lowerbound = min(unscaled_lowerbound, scaled_lowerbound)
        upperbound = max(unscaled_upperbound, scaled_upperbound)

        # Create RvE plot -- bin counts
        fig = plt.figure()
        ax = fig.add_subplot(5, 10, (2,10))
        ax.set_title('Binned RvE Plot -- {}'.format(title))
        # ax.set_xlabel('model error estimates / standard deviation')
        ax.set_xticks([])
        # ax_set_xticklabels([])
        ax.set_ylabel('bin counts')
        # unscaled
        ax.hist(unscaled_model_errors, bins=unscaled_number_of_bins, color='grey', edgecolor='black', alpha=0.5,
                label='uncalibrated')
        # scaled
        ax.hist(scaled_model_errors, bins=scaled_number_of_bins, color='blue', edgecolor='black', alpha=0.5,
                label='calibrated')
        ax.legend(loc="upper right")

        # Create RvE plot
        ax = fig.add_subplot(5,10,(12,40))
        x = np.linspace(lowerbound, upperbound, 100)
        ax.plot(x, x, color='red', label='identity function', alpha=0.5, linestyle='dashed')
        #ax.set_title('Binned RvE Plot -- {}'.format(title))
        ax.set_xlabel('uncertainty estimates')
        ax.set_ylabel('RMS residuals')
        # unscaled stuff
        ax.plot(unscaled_binned_model_errors, unscaled_RMS_abs_res, 'o', color='grey', alpha=0.5)
        ax.plot(unscaled_xfit, unscaled_yfit, color='grey', label='fitted line (uncalibrated)', alpha=0.5)
        #ax.text(0.04, 0.92, '$R^2$ = %.3f' % (r_squared), transform=ax.transAxes)
        #ax.text(0.04, 0.85, 'slope = %.3f' % (slope), transform=ax.transAxes)
        #ax.text(0.04, 0.78, 'y-intercept = %.3f' % (intercept), transform=ax.transAxes)
        # scaled stuff
        ax.plot(scaled_binned_model_errors, scaled_RMS_abs_res, 'o', color='blue', alpha=0.5)
        ax.plot(scaled_xfit, scaled_yfit, color='blue', label='fitted line (calibrated)', alpha=0.5)
        #ax.text(0.04, 0.92, '$R^2$ = %.3f' % (r_squared), transform=ax.transAxes)
        #ax.text(0.04, 0.85, 'slope = %.3f' % (slope), transform=ax.transAxes)
        #ax.text(0.04, 0.78, 'y-intercept = %.3f' % (intercept), transform=ax.transAxes)

        ax.legend(loc="upper left")

        # Add table
        colLabels = ['fitted slope', 'fitted y-intercept', '$R^2$']
        rowLabels = ['uncalibrated', 'calibrated']
        unscaled = ['%.3f' % (unscaled_slope),
                    '%.3f' % (unscaled_intercept), '%.3f' % (unscaled_r_squared)]
        scaled = ['%.3f' % (scaled_slope),
                    '%.3f' % (scaled_intercept), '%.3f' % (scaled_r_squared)]
        cellText = [unscaled, scaled]
        ax.table(cellText=cellText, rowLoc='center', rowLabels=rowLabels, colWidths=[1/3, 1/3, 1/3], colLabels=colLabels,
                 colLoc='center', loc='bottom', bbox=[0, -0.57, 1, 0.3])


        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return unscaled_model_errors, abs_res, unscaled_r_squared, unscaled_slope, unscaled_intercept, unscaled_binned_model_errors, unscaled_RMS_abs_res, unscaled_xfit, unscaled_yfit, scaled_model_errors, abs_res, scaled_r_squared, scaled_slope, scaled_intercept, scaled_binned_model_errors, scaled_RMS_abs_res, scaled_xfit, scaled_yfit

    def make_rstat_table(self, residuals, unscaled_model_errors, scaled_model_errors, title, save=False, file_name=None):
        # Eliminate model errors with value 0, so that the ratios can be calculated
        zero_indices = []
        for i in range(0, len(unscaled_model_errors)):
            if unscaled_model_errors[i] == 0:
                zero_indices.append(i)
        unscaled_residuals = np.delete(residuals, zero_indices)
        unscaled_model_errors = np.delete(unscaled_model_errors, zero_indices)
        print("{} values deleted before making r-stat plot because unscaled model errors were zero.".format(len(zero_indices)))
        scaled_zero_indices = []
        for i in range(0, len(scaled_model_errors)):
            if scaled_model_errors[i] == 0:
                scaled_zero_indices.append(i)
        scaled_residuals = np.delete(residuals, scaled_zero_indices)
        scaled_model_errors = np.delete(scaled_model_errors, scaled_zero_indices)
        print("{} values deleted before making r-stat plot because scaled model errors were zero.".format(len(zero_indices)))
        # make data for gaussian plot
        gaussian_x = np.linspace(-5, 5, 1000)
        # Add table
        colLabels = ['mean', 'standard deviation']
        rowLabels = ['uncalibrated', 'calibrated']
        unscaled = ['%.3f' % (np.mean(unscaled_residuals / unscaled_model_errors)), '%.3f' % (np.std(unscaled_residuals / unscaled_model_errors))]
        scaled = ['%.3f' % (np.mean(scaled_residuals / scaled_model_errors)), '%.3f' % (np.std(scaled_residuals / scaled_model_errors))]
        cellText = [unscaled, scaled]
        ax.table(cellText=cellText, rowLoc='center', rowLabels=rowLabels, colWidths=[.5, .5], colLabels=colLabels, colLoc='center', loc='bottom', bbox=[0, -0.4, 1, 0.2])
        if save is False:
            plt.show()
        elif save is True:
            if file_name is None:
                print("save is set to True, but no file path specified")
            else:
                plt.savefig(file_name, dpi=300)
        plt.close(fig)
        return 0