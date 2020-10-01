from package import ConvergenceData as cd
import numpy as np



# Load data
X_train = np.load('friedman_500_data/training_x_values.npy')
y_train = np.load('friedman_500_data/training_y_values.npy')


CD = cd.ConvergenceData()
#a_direct, b_direct, r_squared_direct, a_direct_unscaled, b_direct_unscaled, r_squared_direct_unscaled,\
			#a_res_v_err, b_res_v_err, r_squared_res_v_err = CD.all([50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000], "RF", X_train, y_train, num_averaged=10)

a_nll, b_nll = CD.nll([50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000], "RF", X_train, y_train, num_averaged=10)

np.save('friedman_500_data/a_nll.npy', np.asarray(a_nll))
np.save('friedman_500_data/b_nll.npy', np.asarray(b_nll))

#np.save('friedman_500_data/a_direct.npy', np.asarray(a_direct))
#np.save('friedman_500_data/b_direct.npy', np.asarray(b_direct))
#np.save('friedman_500_data/r_squared_direct.npy', np.asarray(r_squared_direct))
#np.save('friedman_500_data/a_direct_unscaled.npy', np.asarray(a_direct_unscaled))
#np.save('friedman_500_data/b_direct_unscaled.npy', np.asarray(b_direct_unscaled))
#np.save('friedman_500_data/r_squared_direct_unscaled.npy', np.asarray(r_squared_direct_unscaled))
#np.save('friedman_500_data/a_rve.npy', np.asarray(a_res_v_err))
#np.save('friedman_500_data/b_rve.npy', np.asarray(b_res_v_err))
#np.save('friedman_500_data/r_squared_rve.npy', np.asarray(r_squared_res_v_err))
