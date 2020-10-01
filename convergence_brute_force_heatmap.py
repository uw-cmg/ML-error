import matplotlib.pyplot as plt
import numpy as np
from package import CVData as cvd
from package import MakePlot as mp


# Load friedman data
X_train = np.load('friedman_500_data/training_x_values.npy')
y_train = np.load('friedman_500_data/training_y_values.npy')

# Load diffusion data
#X_train = np.load('diffusion_data/all_x_values.npy')
#y_train = np.load('diffusion_data/all_y_values.npy')
stdev = np.std(y_train)

# Get CV residuals and model errors
CVD = cvd.CVData()
CV_residuals, CV_model_errors = CVD.get_residuals_and_model_errors("RF", X_train, y_train)

# Divide by data set standard deviation
CV_residuals = CV_residuals / stdev
CV_model_errors = CV_model_errors /stdev

#plt.hist(CV_residuals)
#plt.xlabel("Friedman CV residuals / dataset stdev")
#plt.title("Friedman CV residual distribution")
#plt.show()

print("residuals mean:")
print(np.mean(CV_residuals))
print("residuals stdev:")
print(np.std(CV_residuals))

def rstat_stats(x,y):
    ratio = CV_residuals / (CV_model_errors * x + y)
    mu = np.mean(ratio)
    sigma = np.std(ratio)
    third_moment = mu * (mu**2 + 3*sigma**2)
    fourth_moment = mu**4 + 6 * mu**2 * sigma**2 + 3 * sigma**4
    return np.log(mu**2 + (sigma - 1)**2 + third_moment**2 + (fourth_moment - 3)**2)

def nll(x, y):
    sum = 0
    for i in range(0, len(CV_residuals)):
        sum += np.log(2 * np.pi) + np.log((x * CV_model_errors[i] + y)**2) + (CV_residuals[i])**2 / (x * CV_model_errors[i] + y)**2
    return 0.5 * sum / len(CV_residuals)


# generate 2 2d grids for the x & y bounds
a_list = np.linspace(0.1, 1.0, 91)
b_list = np.linspace(0.01, 0.3, 30)

a, b = np.meshgrid(a_list, b_list, indexing='ij')


#z = (1 - a / 2. + a ** 5 + b ** 3) * np.exp(-a ** 2 - b ** 2)

z = a * 0.0 + b * 0.0

for i in range(0, len(a_list)):
    for j in range(0, len(b_list)):
        z[i, j] = nll(a[i, j], b[i, j])
        #z[i,j] = rstat_stats(a[i,j], b[i,j])
        if z[i,j] >= -0.1:
            z[i,j] = -0.1

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = z.min(), z.max()

fig, ax = plt.subplots()

c = ax.pcolormesh(a, b, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('NLL of friedman error estimates scaled with a,b')
fig.colorbar(c, ax=ax)
# set the limits of the plot to the limits of the data
ax.axis([a.min(), a.max(), b.min(), b.max()])
#fig.colorbar(c, ax=ax)
ax.set_xlabel('a (slope)')
ax.set_ylabel('b (intercept)')

plt.show()