# This script generates an array of synthetic training data, trains rf and gpr models
# on it, and saves both the data and the models.

import statistics
import numpy as np
from numpy import arange, meshgrid, array, round
from sklearn.model_selection import ShuffleSplit
from package import rf
from package import CorrectionFactors as cf 
from package import CVData as cvd
from package import ConvergenceData as cd
from package import MakePlot as mp
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

# Define training data size
training_num = 500
# Define noise scale factor
scale = 0

# Define distribution of x-values:
x0_train=np.random.rand(training_num)*0.5
x1_train=np.random.rand(training_num)*0.5
x2_train=np.random.rand(training_num)*0.5
x3_train=np.random.rand(training_num)*0.5
x4_train=np.random.rand(training_num)*0.5

# Put x-values together in a single array
X_train = [[x0_train[i], x1_train[i], x2_train[i], x3_train[i], x4_train[i]] for i in range(0,training_num)]
X_train = np.asarray(X_train)

# Define y-values with synthetic function
y_train = 30*np.sin(4*np.pi*x0_train*x1_train) + 20*(x2_train - 0.5)**2 + 10*x3_train + 5*x4_train

# Define standard deviation of training data
standard_deviation = np.std(y_train)

#generate noise
mu = 0
sigma = standard_deviation * scale
y_noise = np.random.normal(mu, sigma, len(y_train))

# add noise to y-values
y_train = y_train + y_noise

################# initial test plots ####################

#CVD = cvd.CVData()
#residuals, model_errors = CVD.get_residuals_and_model_errors("RF", X_train, y_train)
#residuals = residuals / standard_deviation
#model_errors = model_errors / standard_deviation

#MP = mp.MakePlot()
#MP.make_rve(residuals, model_errors, "RF, Unscaled, Synthetic", save=True, file_name='plots/testplot.png')
#MP.make_rve_bin_counts(model_errors, "RF, Unscaled, Synthetic", save=True, file_name='plots/testplot_hist.png')
#MP.make_rstat(residuals, model_errors, "RF, Unscaled, Synthetic", save=True, file_name='plots/testplot_rstat.png')

#########################################################

################ convergence test plots #################

CD = cd.ConvergenceData()
a_array, b_array = CD.direct([100, 200, 500], "RF", X_train, y_train, num_averaged=3)
#a_array = [[100,0.4,0.1], [200,0.8,0.05], [500, 0.2, 0.01]]
print(a_array)
print(b_array)
MP = mp.MakePlot()
MP.make_convergence_plot(a_array, "RF, Direct Optimization", "a (slope)", save=True, file_name='plots/testplot_convergence_a.png')
MP.make_convergence_plot(b_array, "RF, Direct Optimization", "b (intercept)", save=True, file_name='plots/testplot_convergence_b.png')

#CF = cf.CorrectionFactors(residuals, model_errors)
#a, b, r_squared = CF.direct()
#print("unscaled direct opt:")
#print(a)
#print(b)
#print(r_squared)

#residuals = residuals / standard_deviation
#model_errors = model_errors / standard_deviation
#CF = cf.CorrectionFactors(residuals, model_errors)
#a, b, r_squared = CF.direct()

#print("scaled direct opt:")
#print(a)
#print(b)
#print(r_squared)

#aprime, bprime, rsquared = CF.rve()
#print(aprime)
#print(bprime)
#print(rsquared)

#CD = cd.ConvergenceData()
#a_array, b_array = CD.direct([100, 200, 300, 400, 500], "RF", X_train, y_train, num_averaged=10)
#print(a_array)
#print(b_array)
