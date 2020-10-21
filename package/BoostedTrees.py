import statistics

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class BoostedTrees:
    bt = None
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
        self.bt = GradientBoostingRegressor(n_estimators=model_num, max_depth=3, min_samples_leaf=1).fit(self.X_train,
                                                                                                self.y_train)

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.transform(x_test)
        if retstd is False:
            return self.bt.predict(x_pred)
        error = []
        preds = []
        for x in range(len(x_pred)):
            preds = []
            #list_of_estimators = list(self.bt.estimators_)
            for pred in self.bt.estimators_:
                preds.append(pred[0].predict([x_pred[x]])[0])
            error.append(statistics.stdev(preds))
        error = np.array(error)
        return self.bt.predict(x_pred), error
