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


	def nll(self, num_models, model_type, X_train, y_train, num_averaged):
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

