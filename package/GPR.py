import statistics
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern, ConstantKernel, WhiteKernel, RBF)
from sklearn.preprocessing import StandardScaler


class GPR:
    gpr = None
    kernel = None
    sc = None
    X_train = None
    y_train = None

    def __init__(self):
        pass

    def train(self, X_train, y_train, model_num):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        #self.kernel = ConstantKernel()*RBF()
        self.kernel = ConstantKernel() + 1.0 ** 2 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)
        base_model = GaussianProcessRegressor(kernel=self.kernel, alpha=0.00001, n_restarts_optimizer=30, normalize_y=False).fit(self.X_train, self.y_train)
        self.gpr = BaggingRegressor(base_estimator=base_model, n_estimators=model_num).fit(self.X_train,
                                                                                                self.y_train)

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.transform(x_test)
        if retstd is False:
            return self.gpr.predict(x_pred)
        error = []
        for x in range(len(x_pred)):
            preds = []
            for pred in self.gpr.estimators_:
                preds.append(pred.predict([x_pred[x]])[0])
            error.append(statistics.stdev(preds))
        error = np.array(error)
        return self.gpr.predict(x_pred), error