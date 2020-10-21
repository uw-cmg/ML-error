import numpy as np
from package import CVData as cvd
from package import CorrectionFactors as cf
from package import MakePlot as mp
from package import TestData as td
from sklearn.model_selection import train_test_split

# Data set used
dataset = "Diffusion"

# Model used
model = "LR"

# Path to save files
path = 'Supplemental_Info/{}/5-Fold/{}'.format(dataset, model)
#path = 'plots/'

# Load data
# X_train = np.load('perovskite_data/all_x_values.npy')
# y_train = np.load('perovskite_data/all_y_values.npy')
X_train = np.load('diffusion_data/all_x_values.npy')
y_train = np.load('diffusion_data/all_y_values.npy')

# Cut down to just 80% of the data to make CV graphs for a single split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=91936274)

# Get CV residuals and model errors
CVD = cvd.CVData()
CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors(model, X_train_split, y_train_split)

# Scale residuals and model errors by data set standard deviation
stdev = np.std(y_train_split)
CV_residuals = CV_residuals / stdev
CV_model_errors = CV_model_errors / stdev

# Get correction factors
CF = cf.CorrectionFactors(CV_residuals, CV_model_errors)
a, b, r_squared = CF.nll()
print('Correction Factors:')
print('a: ' + str(a))
print('b: ' + str(b))
print('r^2: ' + str(r_squared))

# Make scaled and unscaled CV plots
MP = mp.MakePlot()
#unscaled
MP.make_rve(CV_residuals, CV_model_errors, "{}, {}, Unscaled".format(model, dataset), save=True,
            file_name=path + '/CV_Plots/unscaled_RvE.png')
MP.make_rve_bin_counts(CV_model_errors, "{}, {}, Unscaled".format(model, dataset), save=True,
            file_name=path + '/CV_Plots/unscaled_RvE_bin_counts.png')
MP.make_rstat(CV_residuals, CV_model_errors, "{}, {}, Unscaled".format(model, dataset), save=True,
            file_name=path + '/CV_Plots/unscaled_rstat.png')
#scaled
MP.make_rve(CV_residuals, CV_model_errors*a + b, "{}, {}, Scaled".format(model, dataset), save=True,
            file_name=path + '/CV_Plots/scaled_RvE.png')
MP.make_rve_bin_counts(CV_model_errors*a + b, "{}, {}, Scaled".format(model, dataset), save=True,
            file_name=path + '/CV_Plots/scaled_RvE_bin_counts.png')
MP.make_rstat(CV_residuals, CV_model_errors*a + b, "{}, {}, Scaled".format(model, dataset), save=True,
            file_name=path + '/CV_Plots/scaled_rstat.png')


# Get test data residuals and model errors
TD = td.TestData()
Test_residuals, Test_model_errors_unscaled, Test_model_errors_scaled = TD.get_residuals_and_model_errors_looped(model, X_train, y_train)


# Make scaled and unscaled test data plots
MP.make_rve(Test_residuals, Test_model_errors_unscaled, "{}, {}, Unscaled, Test Set".format(model, dataset), save=True,
            file_name=path + '/Test_Plots/unscaled_RvE.png')
MP.make_rve_bin_counts(Test_model_errors_unscaled, "{}, {}, Unscaled, Test Set".format(model, dataset), save=True,
            file_name=path + '/Test_Plots/unscaled_RvE_bin_counts.png')
MP.make_rstat(Test_residuals, Test_model_errors_unscaled, "{}, {}, Unscaled, Test Set".format(model, dataset), save=True,
            file_name=path + '/Test_Plots/unscaled_rstat.png')
#scaled
MP.make_rve(Test_residuals, Test_model_errors_scaled, "{}, {}, Scaled, Test Set".format(model, dataset), save=True,
            file_name=path + '/Test_Plots/scaled_RvE.png')
MP.make_rve_bin_counts(Test_model_errors_scaled, "{}, {}, Scaled, Test Set".format(model, dataset), save=True,
            file_name=path + '/Test_Plots/scaled_RvE_bin_counts.png')
MP.make_rstat(Test_residuals, Test_model_errors_scaled, "{}, {}, Scaled, Test Set".format(model, dataset), save=True,
            file_name=path + '/Test_Plots/scaled_rstat.png')