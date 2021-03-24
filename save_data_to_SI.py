import numpy as np

noise_level = "0.2"

# Load data
X_train = np.load('friedman_500_data/training_x_values.npy')
y_train = np.load('friedman_500_data/training_y_values_{}_noise.npy'.format(noise_level))

combined = []
for i in range(0, len(X_train)):
    curr = np.asarray([X_train[i][0], X_train[i][1], X_train[i][2], X_train[i][3], X_train[i][4], y_train[i]])
    combined.append(curr)

combined = np.asarray(combined)

print(X_train[1])
print(X_train[1][1])
print(y_train[1])

np.savetxt("SI/Noisy_Friedman_500/{}_sigma/training.csv".format(noise_level), combined, header="x0, x1, x2, x3, x4, y", delimiter=",")