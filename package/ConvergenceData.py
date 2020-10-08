from package import rf
from package import CorrectionFactors as cf 
from package import CVData as cvd
import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random

class ConvergenceData:

	def __init__(self):
		pass

	def direct(self, num_models, model_type, X_train, y_train, num_averaged=10):
		if model_type == "RF":
			return self._direct_RF(num_models, model_type, X_train, y_train, num_averaged)
		else:
			print("No valid model_type provided for the 'direct' method in ConvergenceData.")

	def nll(self, num_models, model_type, X_train, y_train, num_averaged=10):
		if model_type == "RF":
			return self._nll_RF(num_models, model_type, X_train, y_train, num_averaged)
		else:
			print("No valid model_type provided for the 'nll' method in ConvergenceData.")

	def all(self, num_models, model_type, X_train, y_train, num_averaged=10):
		if model_type == "RF":
			return self._all_RF(num_models, model_type, X_train, y_train, num_averaged)
		else:
			print("No valid model_type provided for the 'direct' method in ConvergenceData.")


########################### helper functions ##########################

	def _direct_RF(self, num_models, model_type, X_train, y_train, num_averaged):
		stdev = np.std(y_train)
		a_direct = []
		b_direct = []
		for i in range(0, len(num_models)):
			a = np.asarray([])
			b = np.asarray([])
			for j in range(0, num_averaged):
				print("i: {}, j: {}".format(i,j))
				# Get residuals and model errors from a set of CV split
				CVD = cvd.CVData()
				seedValue = random.randint(1000000, 10000000)
				residuals, model_errors = CVD.get_residuals_and_model_errors(model_type, \
					X_train, y_train, model_num=num_models[i], random_state=seedValue)
				residuals = residuals / stdev
				model_errors = model_errors / stdev
				# Get correction factors for this CV data
				CF = cf.CorrectionFactors(residuals, model_errors)
				a_curr, b_curr, r_squared = CF.direct()
				a = np.append(a, a_curr)
				b = np.append(b, b_curr)
			# Calculate average for the current number of trees
			a_mu = np.mean(a)
			a_sigma = np.std(a)
			b_mu = np.mean(b)
			b_sigma = np.std(b)
			a_curr = [num_models[i], a_mu, a_sigma]
			b_curr = [num_models[i], b_mu, b_sigma]
			a_direct.append(a_curr)
			b_direct.append(b_curr)
		return a_direct, b_direct


	def _nll_RF(self, num_models, model_type, X_train, y_train, num_averaged):
		stdev = np.std(y_train)
		a_nll = []
		b_nll = []
		for i in range(0, len(num_models)):
			a = np.asarray([])
			b = np.asarray([])
			for j in range(0, num_averaged):
				print("i: {}, j: {}".format(i,j))
				# Get residuals and model errors from a set of CV split
				CVD = cvd.CVData()
				seedValue = random.randint(1000000, 10000000)
				residuals, model_errors = CVD.get_residuals_and_model_errors(model_type, \
					X_train, y_train, model_num=num_models[i], random_state=seedValue)
				residuals = residuals / stdev
				model_errors = model_errors / stdev
				# Get correction factors for this CV data
				CF = cf.CorrectionFactors(residuals, model_errors)
				a_curr, b_curr, r_squared = CF.nll()
				a = np.append(a, a_curr)
				b = np.append(b, b_curr)
			# Calculate average for the current number of trees
			a_mu = np.mean(a)
			a_sigma = np.std(a)
			b_mu = np.mean(b)
			b_sigma = np.std(b)
			a_curr = [num_models[i], a_mu, a_sigma]
			b_curr = [num_models[i], b_mu, b_sigma]
			a_nll.append(a_curr)
			b_nll.append(b_curr)
		return a_nll, b_nll

	def _all_RF(self, num_models, model_type, X_train, y_train, num_averaged):
		stdev = np.std(y_train)
		a_direct = []
		b_direct = []
		r_squared_direct = []
		a_direct_unscaled = []
		b_direct_unscaled = []
		r_squared_direct_unscaled = []
		a_res_v_err = []
		b_res_v_err = []
		r_squared_res_v_err = []
		for i in range(0, len(num_models)):
			a_d = np.asarray([])
			b_d = np.asarray([])
			r_d = np.asarray([])
			a_u = np.asarray([])
			b_u = np.asarray([])
			r_u = np.asarray([])
			a_rve = np.asarray([])
			b_rve = np.asarray([])
			r_rve = np.asarray([])
			for j in range(0, num_averaged):
				print("i: {}, j: {}".format(i,j))
				# Get residuals and model errors from a set of CV split
				CVD = cvd.CVData()
				seedValue = random.randint(1000000, 10000000)
				residuals, model_errors = CVD.get_residuals_and_model_errors(model_type, \
					X_train, y_train, model_num=num_models[i], random_state=seedValue)

				#unscaled direct
				CF = cf.CorrectionFactors(residuals, model_errors)
				a_curr, b_curr, r_squared = CF.direct()
				b_curr = b_curr / stdev
				a_u = np.append(a_u, a_curr)
				b_u = np.append(b_u, b_curr)
				r_u = np.append(r_u, r_squared)

				#scaled direct
				residuals = residuals / stdev
				model_errors = model_errors / stdev
				# Get correction factors for this CV data
				CF = cf.CorrectionFactors(residuals, model_errors)
				a_curr, b_curr, r_squared = CF.direct()
				a_d = np.append(a_d, a_curr)
				b_d = np.append(b_d, b_curr)
				r_d = np.append(r_d, r_squared)

				#RvE
				a_curr, b_curr, r_squared = CF.rve()
				a_rve = np.append(a_rve, a_curr)
				b_rve = np.append(b_rve, b_curr)
				r_rve = np.append(r_rve, r_squared)

			# Calculate average for the current number of trees
			#unscaled direct
			a_mu_u = np.mean(a_u)
			a_sigma_u = np.std(a_u)
			b_mu_u = np.mean(b_u)
			b_sigma_u = np.std(b_u)
			r_squared_mu_u = np.mean(r_u)
			r_squared_sigma_u = np.std(r_u)
			a_curr = [num_models[i], a_mu_u, a_sigma_u]
			b_curr = [num_models[i], b_mu_u, b_sigma_u]
			r_curr = [num_models[i], r_squared_mu_u, r_squared_sigma_u]
			a_direct_unscaled.append(a_curr)
			b_direct_unscaled.append(b_curr)
			r_squared_direct_unscaled.append(r_curr)

			# scaled direct
			a_mu_d = np.mean(a_d)
			a_sigma_d = np.std(a_d)
			b_mu_d = np.mean(b_d)
			b_sigma_d = np.std(b_d)
			r_squared_mu_d = np.mean(r_d)
			r_squared_sigma_d = np.std(r_d)
			a_curr = [num_models[i], a_mu_d, a_sigma_d]
			b_curr = [num_models[i], b_mu_d, b_sigma_d]
			r_curr = [num_models[i], r_squared_mu_d, r_squared_sigma_d]
			a_direct.append(a_curr)
			b_direct.append(b_curr)
			r_squared_direct.append(r_curr)

			# rve
			a_mu_rve = np.mean(a_rve)
			a_sigma_rve = np.std(a_rve)
			b_mu_rve = np.mean(b_rve)
			b_sigma_rve = np.std(b_rve)
			r_squared_mu_rve = np.mean(r_rve)
			r_squared_sigma_rve = np.std(r_rve)
			a_curr = [num_models[i], a_mu_rve, a_sigma_rve]
			b_curr = [num_models[i], b_mu_rve, b_sigma_rve]
			r_curr = [num_models[i], r_squared_mu_rve, r_squared_sigma_rve]
			a_res_v_err.append(a_curr)
			b_res_v_err.append(b_curr)
			r_squared_res_v_err.append(r_curr)
		return a_direct, b_direct, r_squared_direct, a_direct_unscaled, b_direct_unscaled, r_squared_direct_unscaled,\
			a_res_v_err, b_res_v_err, r_squared_res_v_err


