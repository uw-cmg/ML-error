import statistics

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class LinReg:
    lr = None
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
        base_model = Ridge(fit_intercept=True)
        self.lr = BaggingRegressor(base_estimator=base_model, n_estimators=model_num).fit(self.X_train,
                                                                                                self.y_train)

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.transform(x_test)
        if retstd is False:
            return self.lr.predict(x_pred)
        error = []
        for x in range(len(x_pred)):
            preds = []
            for pred in self.lr.estimators_:
                preds.append(pred.predict([x_pred[x]])[0])
            error.append(statistics.stdev(preds))
        error = np.array(error)
        return self.lr.predict(x_pred), error