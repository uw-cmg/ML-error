from package import rf
from package import LinReg as lr
from package import BoostedTrees as bt
from package import GPR as gpr
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
		elif model_type == "LR":
			return self._get_LR(X_train, y_train, model_num, random_state)
		elif model_type == "BT":
			return self._get_BT(X_train, y_train, model_num, random_state)
		elif model_type == "GPR":
			return self._get_GPR(X_train, y_train, model_num, random_state)
		elif model_type == "GPR_Bayesian":
			return self._get_GPR_bayes(X_train, y_train, random_state)
		elif model_type == "GPR_Both":
			return self._get_GPR_both(X_train, y_train, model_num, random_state)
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

	def _get_LR(self, X_values, y_values, model_num, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=random_state)
		# LR
		LR_model_errors = np.asarray([])
		LR_resid = np.asarray([])
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			LR = lr.LinReg()
			LR.train(X_train, y_train, model_num)
			lr_pred, LR_errors = LR.predict(X_test, True)
			lr_res = y_test - lr_pred
			LR_model_errors = np.concatenate((LR_model_errors, LR_errors), axis=None)
			LR_resid = np.concatenate((LR_resid, lr_res), axis=None)

		return LR_resid, LR_model_errors

	def _get_BT(self, X_values, y_values, model_num, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=random_state)
		# BT
		model_errors = np.asarray([])
		resid = np.asarray([])
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			BT = bt.BoostedTrees()
			BT.train(X_train, y_train, model_num)
			pred, errors = BT.predict(X_test, True)
			res = y_test - pred
			model_errors = np.concatenate((model_errors, errors), axis=None)
			resid = np.concatenate((resid, res), axis=None)

		return resid, model_errors

	def _get_GPR(self, X_values, y_values, model_num, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=random_state)
		# GPR
		model_errors = np.asarray([])
		resid = np.asarray([])
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			GPR = gpr.GPR()
			GPR.train(X_train, y_train, model_num)
			pred, errors = GPR.predict(X_test, True)
			res = y_test - pred
			model_errors = np.concatenate((model_errors, errors), axis=None)
			resid = np.concatenate((resid, res), axis=None)

		return resid, model_errors

	def _get_GPR_bayes(self, X_values, y_values, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=random_state)
		# GPR
		model_errors = np.asarray([])
		resid = np.asarray([])
		for train_index, test_index in rkf.split(X_values):
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			GPR = gpr.GPR()
			GPR.train_single(X_train, y_train)
			pred, errors = GPR.predict_single(X_test, retstd=True)
			res = y_test - pred
			model_errors = np.concatenate((model_errors, errors), axis=None)
			resid = np.concatenate((resid, res), axis=None)

		return resid, model_errors

	def _get_GPR_both(self, X_values, y_values, model_num, random_state):
		rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=random_state)
		# GPR
		model_errors_bayes = np.asarray([])
		model_errors_bootstrap = np.asarray([])
		resid = np.asarray([])
		i = 1
		for train_index, test_index in rkf.split(X_values):
			print("Starting cross-validation loop {} of 20".format(i))
			i = i + 1
			X_train, X_test = X_values[train_index], X_values[test_index]
			y_train, y_test = y_values[train_index], y_values[test_index]
			# predict with single model
			GPR_bayes = gpr.GPR()
			GPR_bayes.train_single(X_train, y_train)
			pred_bayes, errors_bayes = GPR_bayes.predict_single(X_test, retstd=True)
			# predict with bootstrap ensemble
			GPR_bootstrap = gpr.GPR()
			GPR_bootstrap.train(X_train, y_train, model_num)
			pred_bootstrap, errors_bootstrap = GPR_bootstrap.predict(X_test, True)
			# save the bayes (single model) residuals, and both sets of model errors
			res = y_test - pred_bayes
			model_errors_bayes = np.concatenate((model_errors_bayes, errors_bayes), axis=None)
			model_errors_bootstrap = np.concatenate((model_errors_bootstrap, errors_bootstrap), axis=None)
			resid = np.concatenate((resid, res), axis=None)

		return resid, model_errors_bayes, model_errors_bootstrap
