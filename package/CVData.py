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