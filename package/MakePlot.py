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
        ax.text(0.05, 0.9, 'mean = %f' % (np.mean(residuals / model_errors)), transform=ax.transAxes)
        ax.text(0.05, 0.85, 'std = %f' % (np.std(residuals / model_errors)), transform=ax.transAxes)
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
        ax.set_ylabel('residuals / standard deviation')
        ax.plot(binned_model_errors, RMS_abs_res, 'o', color='blue')
        ax.plot(xfit, yfit, color='blue', label='fitted line')
        ax.text(0.04, 0.92, 'r^2 = %.3f' % (r_squared), transform=ax.transAxes)
        ax.text(0.04, 0.85, 'slope = %.3f' % (slope), transform=ax.transAxes)
        ax.text(0.04, 0.78, 'y-intercept = %.3f' % (intercept), transform=ax.transAxes)
        ax.legend(loc="lower right")

        ax = fig.add_subplot(3,1,1)
        ax.set_title('Binned RvE Plot -- {}'.format(title))
        #ax.set_xlabel('model error estimates / standard deviation')
        ax.set_xticks([])
        #ax_set_xticklabels([])
        ax.set_ylabel('bin counts')
        ax.set_xlim([0.05, 0.8])
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