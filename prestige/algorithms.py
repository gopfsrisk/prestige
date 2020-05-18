import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# linear regression class
class OLSRegression:
	# initialize class
	def __init__(self, fit_intercept=True):
		self.fit_intercept = fit_intercept
	# fit the model
	def fit(self, X, y):
		# if fit_intercept == True
		if self.fit_intercept:
			# generate array of ones
			arr_ones = np.array(np.ones(X.shape[0]))
			# add as col to X
			X.insert(loc=0, column='intercept', value=arr_ones)
		# ordinary least squares b = (XT*X)^-1 * XT*y
		arr_betas = inv(X.T.dot(X)).dot(X.T).dot(y)
		# create dictionary
		dict_col_betas = dict(zip(X.columns, arr_betas))
		# save to class
		self.dict_col_betas = dict_col_betas
		# return self
		return self
	# generate predictions
	def predict(self, X):
		# if fit_intercept == True
		if (self.fit_intercept) and ('intercept' not in list(X.columns)):
			# generate array of ones
			arr_ones = np.array(np.ones(X.shape[0]))
			# add as col to X
			X.insert(loc=0, column='intercept', value=arr_ones)
		# multiply each cell by its beta
		list_predictions = list(X.dot(pd.Series(self.dict_col_betas)))
		# save to class
		self.list_predictions = list_predictions
		# return self
		return self
	# evaluate performance
	def evaluate(self, y):
		# get mean absolute error
		mae = mean_absolute_error(y_true=y, y_pred=self.list_predictions)
		# get mean squared error
		mse = mean_squared_error(y_true=y, y_pred=self.list_predictions)
		# get root mean square error
		rmse = np.sqrt(mse)
		# get explained variance (R2)
		r2 = r2_score(y_true=y, y_pred=self.list_predictions)
		# create dictionary
		dict_eval_metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
		# return dict_eval_metrics
		return dict_eval_metrics