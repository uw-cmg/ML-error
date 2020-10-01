from package import rf

class TestData:

	def __init__(self):
		pass

	def get_residuals_and_model_errors(self, model_type, X_train, y_train, X_test, y_test, model_num=200, random_state=91936274):
		if model_type == "RF":
			return self._get_RF(X_train, y_train, X_test, y_test, model_num, random_state)
		else:
			print("No valid model type was provided in the 'get_residuals_and_model_errors' CVData method.")
		return 0

	def _get_RF(self, X_train, y_train, X_test, y_test, model_num, random_state):
		RF = rf.RF()
		RF.train(X_train, y_train, model_num)
		predictions, model_errors = RF.predict(X_test, True)
		residuals = y_test - predictions
		return residuals, model_errors