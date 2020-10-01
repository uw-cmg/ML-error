import statistics

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RF:
    rf = None
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
        self.rf = RandomForestRegressor(n_estimators=model_num, max_depth=30, min_samples_leaf=1).fit(self.X_train,
                                                                                                self.y_train)

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.transform(x_test)
        if retstd is False:
            return self.rf.predict(x_pred)
        error = []
        preds = []
        for x in range(len(x_pred)):
            preds = []
            for pred in self.rf.estimators_:
                preds.append(pred.predict([x_pred[x]])[0])
            error.append(statistics.stdev(preds))
        error = np.array(error)
        return self.rf.predict(x_pred), error
