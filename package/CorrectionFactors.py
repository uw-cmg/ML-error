from package import rf
from package import CVData as cvd
import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.stats as stats

class CorrectionFactors:
	residuals = None
	model_errors = None

	def __init__(self, residuals, model_errors):
		self.residuals = residuals
		self.model_errors = model_errors

	# Function to find scale factors by directly optimizing the r-stat distribution
	# The r^2 value returned is obtained by making a binned residual vs. error plot and
	# fitting a line, after scaling with the a and b found by this function.
	def direct(self):
		x0 = np.array([1.0, 0.0])
		res = minimize(self._direct_opt, x0, method='nelder-mead')
		a = res.x[0]
		b = res.x[1]
		success = res.success
		if success is True:
			print("r-stat optimization successful!")
		elif success is False:
			print("r-stat optimization failed.")
		#print(res)
		r_squared = self._direct_rsquared(a,b)
		return a, b, r_squared

	def nll(self):
		x0 = np.array([1.0, 0.0])
		res = minimize(self._nll_opt, x0, method='nelder-mead')
		a = res.x[0]
		b = res.x[1]
		success = res.success
		if success is True:
			print("NLL optimization successful!")
		elif success is False:
			print("NLL optimization failed.")
		# print(res)
		r_squared = self._direct_rsquared(a, b)
		return a, b, r_squared

	# Function to find scale factors using binned residual vs. model error plot
	def rve(self, number_of_bins=15):
		model_errors = self.model_errors
		abs_res = abs(self.residuals)

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
		for i in range(1,number_of_bins + 1):
		    if i in digitized:
		        weights.append(np.count_nonzero(digitized == i))
    	
		# Calculate RMS of the absolute residuals
		RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in range(0, len(bins_present))]

		# Set the x-values to the midpoint of each bin
		bin_width = bins[1]-bins[0]
		binned_model_errors = np.zeros(len(bins_present))
		for i in range(0, len(bins_present)):
		    curr_bin = bins_present[i]
		    binned_model_errors[i] = bins[curr_bin-1] + bin_width/2

		# Fit a line to the data
		model = LinearRegression(fit_intercept=True)
		model.fit(binned_model_errors[:, np.newaxis],
		              RMS_abs_res, sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
		xfit = binned_model_errors
		yfit = model.predict(xfit[:, np.newaxis])

		# Calculate r^2 value
		r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
		# Calculate slope
		slope = model.coef_
		# Calculate y-intercept
		intercept = model.intercept_

		#print("rf slope: {}".format(slope))
		#print("rf intercept: {}".format(intercept))
    	
		return slope, intercept, r_squared


################################## Helper functions #######################################


	def _direct_opt(self, x):
		ratio = self.residuals / (self.model_errors * x[0] + x[1])
		sigma = np.std(ratio)
		mu = np.mean(ratio)
		return mu**2 + (sigma - 1)**2

	def _nll_opt(self, x):
		sum = 0
		for i in range(0, len(self.residuals)):
			sum += np.log(2 * np.pi) + np.log((x[0] * self.model_errors[i] + x[1]) ** 2) + (self.residuals[i]) ** 2 / (
						x[0] * self.model_errors[i] + x[1]) ** 2
		return 0.5 * sum / len(self.residuals)

	def _direct_rsquared(self, a, b, number_of_bins=15):
		model_errors = self.model_errors * a + b
		abs_res = abs(self.residuals)

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
		for i in range(1,number_of_bins + 1):
		    if i in digitized:
		        weights.append(np.count_nonzero(digitized == i))
    	
		# Calculate RMS of the absolute residuals
		RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in range(0, len(bins_present))]

		# Set the x-values to the midpoint of each bin
		bin_width = bins[1]-bins[0]
		binned_model_errors = np.zeros(len(bins_present))
		for i in range(0, len(bins_present)):
		    curr_bin = bins_present[i]
		    binned_model_errors[i] = bins[curr_bin-1] + bin_width/2

		# Fit a line to the data
		model = LinearRegression(fit_intercept=True)
		model.fit(binned_model_errors[:, np.newaxis],
		              RMS_abs_res, sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
		xfit = binned_model_errors
		yfit = model.predict(xfit[:, np.newaxis])

		# Calculate r^2 value
		r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
		# Calculate slope
		slope = model.coef_
		# Calculate y-intercept
		intercept = model.intercept_

		#print("rf slope: {}".format(slope))
		#print("rf intercept: {}".format(intercept))
    	
		return r_squared


		

