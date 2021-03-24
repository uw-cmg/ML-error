import numpy as np

# Define training data size
training_num = 500
# Define noise scale factor
scale = 0.4
# Load friedman no-noise training set
y_train = np.load('friedman_500_data/training_y_values.npy')
# Load friedman no-noise test set
y_test = np.load('friedman_500_data/test_y_values_hypercube.npy')

# Define standard deviation of training data
standard_deviation = np.std(y_train)

#generate noise
mu = 0
sigma = standard_deviation * scale
y_noise_train = np.random.normal(mu, sigma, len(y_train))
y_noise_test = np.random.normal(mu, sigma, len(y_test))

print(np.mean(y_noise_train))
print(np.std(y_noise_train))
print(np.std(y_train))

# add noise to y-values
y_train = y_train + y_noise_train
y_test = y_test + y_noise_test

np.save('friedman_500_data/training_y_values_{}_noise.npy'.format(scale), y_train)
np.save('friedman_500_data/test_y_values_{}_noise.npy'.format(scale), y_test)