from package import rf
import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class CVData:

	def __init__(self):
		pass

	def get_residuals_and_model_errors(self, model_type, X_train, y_train, model_num=200, random_state=91936274):
		if model_type == "RF":
			return self._get_RF(X_train, y_train, model_num, random_state)
		else:
			print("No valid model type was provided in the 'get_residuals_and_model_errors' CVData method.")

	def _get_RF(self, X_values, y_values, model_num, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=random_state)
		# RF
		RF_model_errors = np.asarray([])
		RF_resid = np.asarray([])
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			RF = rf.RF()
			RF.train(X_train, y_train, model_num)
			rf_pred, RF_errors = RF.predict(X_test, True)
			rf_res = y_test - rf_pred
			RF_model_errors = np.concatenate((RF_model_errors, RF_errors), axis=None)
			RF_resid = np.concatenate((RF_resid, rf_res), axis=None)

		return RF_resid, RF_model_errors