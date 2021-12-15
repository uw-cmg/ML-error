from package import rf
from package import CVData as cvd
from package import CorrectionFactors as cf
from package import LinReg as lr
from package import BoostedTrees as bt
from package import GPR as gpr
from sklearn.model_selection import RepeatedKFold
import numpy as np

class TestData:

	def __init__(self):
		pass

	def get_residuals_and_model_errors(self, model_type, X_train, y_train, X_test, y_test, model_num=200, random_state=91936274):
		if model_type == "RF":
			return self._get_RF(X_train, y_train, X_test, y_test, model_num)
		elif model_type == "LR":
			return self._get_LR(X_train, y_train, X_test, y_test, model_num)
		elif model_type == "BT":
			return self._get_BT(X_train, y_train, X_test, y_test, model_num)
		elif model_type == "GPR":
			return self._get_GPR(X_train, y_train, X_test, y_test, model_num)
		elif model_type == "GPR_Bayesian":
			return self._get_GPR_bayes(X_train, y_train, X_test, y_test)
		elif model_type == "GPR_Both":
			return self._get_GPR_both(X_train, y_train, X_test, y_test, model_num)
		else:
			print("No valid model type was provided in the 'get_residuals_and_model_errors' TestData method.")
		return 0

	def get_residuals_and_model_errors_looped(self, dataset, model_type, X_train, y_train, model_num=200, random_state=91936274):
		if model_type == "RF":
			return self._get_RF_looped(dataset, X_train, y_train, model_num, random_state)
		elif model_type == "LR":
			return self._get_LR_looped(dataset, X_train, y_train, model_num, random_state)
		elif model_type == "BT":
			return self._get_BT_looped(dataset, X_train, y_train, model_num, random_state)
		elif model_type == "GPR":
			return self._get_GPR_looped(dataset, X_train, y_train, model_num, random_state)
		elif model_type == "GPR_Bayesian":
			return self._get_GPR_looped_bayes(dataset, X_train, y_train, random_state)
		elif model_type == "GPR_Both":
			return self._get_GPR_looped_both(dataset, X_train, y_train, model_num, random_state)
		else:
			print("No valid model type was provided in the 'get_residuals_and_model_errors_looped' TestData method.")
		return 0

	def _get_RF(self, X_train, y_train, X_test, y_test, model_num):
		RF = rf.RF()
		RF.train(X_train, y_train, model_num)
		predictions, model_errors = RF.predict(X_test, True)
		residuals = y_test - predictions
		return residuals, model_errors

	def _get_RF_looped(self, dataset, X_values, y_values, model_num, random_state):
		if dataset == "Diffusion":
			rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
		elif dataset == "Perovskite":
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
		else:
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
			print("Neither 'Diffusion' nor 'Perovskite' was specified as the dataset for get_residuals_and_model_errors_looped function.")
			print("Setting repeated k-fold to 5-fold splits repeated twice.")
		# RF
		RF_unscaled_model_errors = np.asarray([])
		RF_scaled_model_errors = np.asarray([])
		RF_resid = np.asarray([])
		a_array = []
		b_array = []
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# Get CV residuals and model errors
			CVD = cvd.CVData()
			CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors("RF", X_train, y_train)
			# Scale residuals and model errors by data set standard deviation
			stdev = np.std(y_train)
			CV_residuals = CV_residuals / stdev
			CV_model_errors = CV_model_errors / stdev
			# Get correction factors
			CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
			a, b, r_squared = CF.nll()
			print('Correction Factors:')
			print('a: ' + str(a))
			print('b: ' + str(b))
			print('r^2: ' + str(r_squared))
			# Record the newly calculated a and b values
			a_array.append(a)
			b_array.append(b)
			# Get test data residuals and model errors
			Test_residuals, Test_model_errors = self._get_RF(X_train, y_train, X_test, y_test, model_num)
			# Scale by standard deviation
			Test_residuals = Test_residuals / stdev
			Test_model_errors = Test_model_errors / stdev
			# Scale model errors using scale factors obtained above
			Test_model_errors_scaled = Test_model_errors * a + b
			# Append results from this split to the arrays to be returned
			RF_unscaled_model_errors = np.concatenate((RF_unscaled_model_errors, Test_model_errors), axis=None)
			RF_scaled_model_errors = np.concatenate((RF_scaled_model_errors, Test_model_errors_scaled), axis=None)
			RF_resid = np.concatenate((RF_resid, Test_residuals), axis=None)
		a_array = np.asarray(a_array)
		b_array = np.asarray(b_array)
		return RF_resid, RF_unscaled_model_errors, RF_scaled_model_errors, a_array, b_array

	def _get_LR(self, X_train, y_train, X_test, y_test, model_num):
		LR = lr.LinReg()
		LR.train(X_train, y_train, model_num)
		predictions, model_errors = LR.predict(X_test, True)
		residuals = y_test - predictions
		return residuals, model_errors

	def _get_LR_looped(self, dataset, X_values, y_values, model_num, random_state):
		if dataset == "Diffusion":
			rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
		elif dataset == "Perovskite":
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
		else:
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
			print("Neither 'Diffusion' nor 'Perovskite' was specified as the dataset for get_residuals_and_model_errors_looped function.")
			print("Setting repeated k-fold to 5-fold splits repeated twice.")
		# LR
		LR_unscaled_model_errors = np.asarray([])
		LR_scaled_model_errors = np.asarray([])
		LR_resid = np.asarray([])
		a_array = []
		b_array = []
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# Get CV residuals and model errors
			CVD = cvd.CVData()
			CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors("LR", X_train, y_train)
			# Scale residuals and model errors by data set standard deviation
			stdev = np.std(y_train)
			CV_residuals = CV_residuals / stdev
			CV_model_errors = CV_model_errors / stdev
			# Get correction factors
			CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
			a, b, r_squared = CF.nll()
			print('Correction Factors:')
			print('a: ' + str(a))
			print('b: ' + str(b))
			print('r^2: ' + str(r_squared))
			# Record the newly calculated a and b values
			a_array.append(a)
			b_array.append(b)
			# Get test data residuals and model errors
			Test_residuals, Test_model_errors = self._get_LR(X_train, y_train, X_test, y_test, model_num)
			# Scale by standard deviation
			Test_residuals = Test_residuals / stdev
			Test_model_errors = Test_model_errors / stdev
			# Scale model errors using scale factors obtained above
			Test_model_errors_scaled = Test_model_errors * a + b
			# Append results from this split to the arrays to be returned
			LR_unscaled_model_errors = np.concatenate((LR_unscaled_model_errors, Test_model_errors), axis=None)
			LR_scaled_model_errors = np.concatenate((LR_scaled_model_errors, Test_model_errors_scaled), axis=None)
			LR_resid = np.concatenate((LR_resid, Test_residuals), axis=None)
		a_array = np.asarray(a_array)
		b_array = np.asarray(b_array)
		return LR_resid, LR_unscaled_model_errors, LR_scaled_model_errors, a_array, b_array

	def _get_BT(self, X_train, y_train, X_test, y_test, model_num):
		BT = bt.BoostedTrees()
		BT.train(X_train, y_train, model_num)
		predictions, model_errors = BT.predict(X_test, True)
		residuals = y_test - predictions
		return residuals, model_errors

	def _get_BT_looped(self, dataset, X_values, y_values, model_num, random_state):
		if dataset == "Diffusion":
			rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
		elif dataset == "Perovskite":
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
		else:
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
			print("Neither 'Diffusion' nor 'Perovskite' was specified as the dataset for get_residuals_and_model_errors_looped function.")
			print("Setting repeated k-fold to 5-fold splits repeated twice.")
		# BT
		unscaled_model_errors = np.asarray([])
		scaled_model_errors = np.asarray([])
		resid = np.asarray([])
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# Get CV residuals and model errors
			CVD = cvd.CVData()
			CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors("BT", X_train, y_train)
			# Scale residuals and model errors by data set standard deviation
			stdev = np.std(y_train)
			CV_residuals = CV_residuals / stdev
			CV_model_errors = CV_model_errors / stdev
			# Get correction factors
			CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
			a, b, r_squared = CF.nll()
			print('Correction Factors:')
			print('a: ' + str(a))
			print('b: ' + str(b))
			print('r^2: ' + str(r_squared))
			# Get test data residuals and model errors
			Test_residuals, Test_model_errors = self._get_BT(X_train, y_train, X_test, y_test, model_num)
			# Scale by standard deviation
			Test_residuals = Test_residuals / stdev
			Test_model_errors = Test_model_errors / stdev
			# Scale model errors using scale factors obtained above
			Test_model_errors_scaled = Test_model_errors * a + b
			# Append results from this split to the arrays to be returned
			unscaled_model_errors = np.concatenate((unscaled_model_errors, Test_model_errors), axis=None)
			scaled_model_errors = np.concatenate((scaled_model_errors, Test_model_errors_scaled), axis=None)
			resid = np.concatenate((resid, Test_residuals), axis=None)
		return resid, unscaled_model_errors, scaled_model_errors

	def _get_GPR(self, X_train, y_train, X_test, y_test, model_num):
		GPR = gpr.GPR()
		GPR.train(X_train, y_train, model_num)
		predictions, model_errors = GPR.predict(X_test, True)
		residuals = y_test - predictions
		return residuals, model_errors

	def _get_GPR_looped(self, dataset, X_values, y_values, model_num, random_state):
		if dataset == "Diffusion":
			rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
		elif dataset == "Perovskite":
			rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
		else:
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
			print("Neither 'Diffusion' nor 'Perovskite' was specified as the dataset for get_residuals_and_model_errors_looped function.")
			print("Setting repeated k-fold to 5-fold splits repeated twice.")
		# GPR
		unscaled_model_errors = np.asarray([])
		scaled_model_errors = np.asarray([])
		resid = np.asarray([])
		a_array = []
		b_array = []
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# Get CV residuals and model errors
			CVD = cvd.CVData()
			CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors("GPR", X_train, y_train)
			# Scale residuals and model errors by data set standard deviation
			stdev = np.std(y_train)
			CV_residuals = CV_residuals / stdev
			CV_model_errors = CV_model_errors / stdev
			# Get correction factors
			CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
			a, b, r_squared = CF.nll()
			print('Correction Factors:')
			print('a: ' + str(a))
			print('b: ' + str(b))
			print('r^2: ' + str(r_squared))
			# Record the newly calculated a and b values
			a_array.append(a)
			b_array.append(b)
			# Get test data residuals and model errors
			Test_residuals, Test_model_errors = self._get_GPR(X_train, y_train, X_test, y_test, model_num)
			# Scale by standard deviation
			Test_residuals = Test_residuals / stdev
			Test_model_errors = Test_model_errors / stdev
			# Scale model errors using scale factors obtained above
			Test_model_errors_scaled = Test_model_errors * a + b
			# Append results from this split to the arrays to be returned
			unscaled_model_errors = np.concatenate((unscaled_model_errors, Test_model_errors), axis=None)
			scaled_model_errors = np.concatenate((scaled_model_errors, Test_model_errors_scaled), axis=None)
			resid = np.concatenate((resid, Test_residuals), axis=None)
		a_array = np.asarray(a_array)
		b_array = np.asarray(b_array)
		return resid, unscaled_model_errors, scaled_model_errors, a_array, b_array

	def _get_GPR_bayes(self, X_train, y_train, X_test, y_test):
		GPR = gpr.GPR()
		GPR.train_single(X_train, y_train)
		predictions, model_errors = GPR.predict_single(X_test, retstd=True)
		residuals = y_test - predictions
		return residuals, model_errors

	def _get_GPR_looped_bayes(self, dataset, X_values, y_values, random_state):
		if dataset == "Diffusion":
			rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
		elif dataset == "Perovskite":
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
		else:
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
			print("Neither 'Diffusion' nor 'Perovskite' was specified as the dataset for get_residuals_and_model_errors_looped function.")
			print("Setting repeated k-fold to 5-fold splits repeated twice.")
		# GPR
		unscaled_model_errors = np.asarray([])
		scaled_model_errors = np.asarray([])
		resid = np.asarray([])
		a_array = []
		b_array = []
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# Get CV residuals and model errors
			CVD = cvd.CVData()
			CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors("GPR_Bayesian", X_train, y_train)
			# Scale residuals and model errors by data set standard deviation
			stdev = np.std(y_train)
			CV_residuals = CV_residuals / stdev
			CV_model_errors = CV_model_errors / stdev
			# Get correction factors
			CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
			a, b, r_squared = CF.nll()
			print('Correction Factors:')
			print('a: ' + str(a))
			print('b: ' + str(b))
			print('r^2: ' + str(r_squared))
			# Record the newly calculated a and b values
			a_array.append(a)
			b_array.append(b)
			# Get test data residuals and model errors
			Test_residuals, Test_model_errors = self._get_GPR_bayes(X_train, y_train, X_test, y_test)
			# Scale by standard deviation
			Test_residuals = Test_residuals / stdev
			Test_model_errors = Test_model_errors / stdev
			# Scale model errors using scale factors obtained above
			Test_model_errors_scaled = Test_model_errors * a + b
			# Append results from this split to the arrays to be returned
			unscaled_model_errors = np.concatenate((unscaled_model_errors, Test_model_errors), axis=None)
			scaled_model_errors = np.concatenate((scaled_model_errors, Test_model_errors_scaled), axis=None)
			resid = np.concatenate((resid, Test_residuals), axis=None)
		a_array = np.asarray(a_array)
		b_array = np.asarray(b_array)
		return resid, unscaled_model_errors, scaled_model_errors, a_array, b_array

	def _get_GPR_both(self, X_train, y_train, X_test, y_test, model_num):
		# predict with bayes (single model)
		GPR_bayes = gpr.GPR()
		GPR_bayes.train_single(X_train, y_train)
		predictions, model_errors_bayes = GPR_bayes.predict_single(X_test, retstd=True)
		residuals = y_test - predictions
		# predict with bootstrap
		GPR_bootstrap = gpr.GPR()
		GPR_bootstrap.train(X_train, y_train, model_num)
		predictions_bootstrap, model_errors_bootstrap = GPR_bootstrap.predict(X_test, True)
		return residuals, model_errors_bayes, model_errors_bootstrap

	def _get_GPR_looped_both(self, dataset, X_values, y_values, model_num, random_state):
		if dataset == "Diffusion":
			rkf = RepeatedKFold(n_splits=5, n_repeats=5)
		elif dataset == "Perovskite":
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
		else:
			rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
			print("Neither 'Diffusion' nor 'Perovskite' was specified as the dataset for get_residuals_and_model_errors_looped function.")
			print("Setting repeated k-fold to 5-fold splits repeated twice.")
		# GPR
		unscaled_model_errors_bayes = np.asarray([])
		scaled_model_errors_bayes = np.asarray([])
		unscaled_model_errors_bootstrap = np.asarray([])
		scaled_model_errors_bootstrap = np.asarray([])
		resid = np.asarray([])
		a_array_bayes = []
		b_array_bayes = []
		a_array_bootstrap = []
		b_array_bootstrap = []
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# Get CV residuals and model errors
			CVD = cvd.CVData()
			CV_residuals, CV_model_errors_bayes, CV_model_errors_bootstrap = CVD.get_residuals_and_model_errors("GPR_Both", X_train, y_train)
			# Scale residuals and model errors by data set standard deviation
			stdev = np.std(y_train)
			CV_residuals = CV_residuals / stdev
			CV_model_errors_bayes = CV_model_errors_bayes / stdev
			CV_model_errors_bootstrap = CV_model_errors_bootstrap / stdev
			# Get correction factors
			CF_bayes = cf.CorrectionFactors(CV_residuals, CV_model_errors_bayes)
			CF_bootstrap = cf.CorrectionFactors(CV_residuals, CV_model_errors_bootstrap)
			a_bayes, b_bayes, r_squared_bayes = CF_bayes.nll()
			a_bootstrap, b_bootstrap, r_squared_bootstrap = CF_bootstrap.nll()
			print('Correction Factors:')
			print('a_bayes: ' + str(a_bayes))
			print('b_bayes: ' + str(b_bayes))
			print('r^2_bayes: ' + str(r_squared_bayes))
			print('a_bootstrap: ' + str(a_bootstrap))
			print('b_bootstrap: ' + str(b_bootstrap))
			print('r^2_bootstrap: ' + str(r_squared_bootstrap))
			# Record the newly calculated a and b values
			a_array_bayes.append(a_bayes)
			b_array_bayes.append(b_bayes)
			a_array_bootstrap.append(a_bootstrap)
			b_array_bootstrap.append(b_bootstrap)
			# Get test data residuals and model errors
			Test_residuals, Test_model_errors_bayes, Test_model_errors_bootstrap = self._get_GPR_both(X_train, y_train, X_test, y_test, model_num)
			# Scale by standard deviation
			Test_residuals = Test_residuals / stdev
			Test_model_errors_bayes = Test_model_errors_bayes / stdev
			Test_model_errors_bootstrap = Test_model_errors_bootstrap / stdev
			# Scale model errors using scale factors obtained above
			Test_model_errors_scaled_bayes = Test_model_errors_bayes * a_bayes + b_bayes
			Test_model_errors_scaled_bootstrap = Test_model_errors_bootstrap * a_bootstrap + b_bootstrap
			# Append results from this split to the arrays to be returned
			unscaled_model_errors_bayes = np.concatenate((unscaled_model_errors_bayes, Test_model_errors_bayes), axis=None)
			scaled_model_errors_bayes = np.concatenate((scaled_model_errors_bayes, Test_model_errors_scaled_bayes), axis=None)
			unscaled_model_errors_bootstrap = np.concatenate((unscaled_model_errors_bootstrap, Test_model_errors_bootstrap), axis=None)
			scaled_model_errors_bootstrap = np.concatenate((scaled_model_errors_bootstrap, Test_model_errors_scaled_bootstrap), axis=None)
			resid = np.concatenate((resid, Test_residuals), axis=None)
		a_array_bayes = np.asarray(a_array_bayes)
		b_array_bayes = np.asarray(b_array_bayes)
		a_array_bootstrap = np.asarray(a_array_bootstrap)
		b_array_bootstrap = np.asarray(b_array_bootstrap)
		return resid, unscaled_model_errors_bayes, scaled_model_errors_bayes, a_array_bayes, b_array_bayes, unscaled_model_errors_bootstrap, scaled_model_errors_bootstrap, a_array_bootstrap, b_array_bootstrap
