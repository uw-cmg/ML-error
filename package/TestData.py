from package import rf
from package import CVData as cvd
from package import CorrectionFactors as cf
from sklearn.model_selection import RepeatedKFold
import numpy as np

class TestData:

	def __init__(self):
		pass

	def get_residuals_and_model_errors(self, model_type, X_train, y_train, X_test, y_test, model_num=200, random_state=91936274):
		if model_type == "RF":
			return self._get_RF(X_train, y_train, X_test, y_test, model_num)
		else:
			print("No valid model type was provided in the 'get_residuals_and_model_errors' CVData method.")
		return 0

	def get_residuals_and_model_errors_looped(self, model_type, X_train, y_train, model_num=200, random_state=91936274):
		if model_type == "RF":
			return self._get_RF_looped(X_train, y_train, model_num, random_state)
		else:
			print("No valid model type was provided in the 'get_residuals_and_model_errors_looped' CVData method.")
		return 0

	def _get_RF(self, X_train, y_train, X_test, y_test, model_num):
		RF = rf.RF()
		RF.train(X_train, y_train, model_num)
		predictions, model_errors = RF.predict(X_test, True)
		residuals = y_test - predictions
		return residuals, model_errors

	def _get_RF_looped(self, X_values, y_values, model_num, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
		# RF
		RF_unscaled_model_errors = np.asarray([])
		RF_scaled_model_errors = np.asarray([])
		RF_resid = np.asarray([])
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
		return RF_resid, RF_unscaled_model_errors, RF_scaled_model_errors
