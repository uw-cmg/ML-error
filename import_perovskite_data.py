# This script imports the PV data set from its CSV file, removes unnecessary columns, and saves the x- and y-values as np arrays.
from package import io
import numpy as np

# import data
#data = io.importdata('perovskite_data/PVstability_Weipaper_alldata_featureselected.csv')
#data = io.sanitizedata(data, user_list=['is_testdata', 'Material Composition'])
data = io.importdata('perovskite_data/Perovskite_stability_Wei_updated.csv')
data = io.sanitizedata(data, user_list=['Compositions'])

# separate x- and y-values and save as numpy arrays
X_values = data.iloc[:, 1:]
y_values = data.iloc[:, 0]
X_values = X_values.to_numpy(dtype=float)
y_values = y_values.to_numpy(dtype=float)

# save arrays for later use
np.save('perovskite_data/all_x_values.npy', X_values)
np.save('perovskite_data/all_y_values.npy', y_values)
