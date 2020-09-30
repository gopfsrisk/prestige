import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import catboost as cb

# linear regression class
class OLSRegression:
	"""
	Takes an X data frame and y array and completes Ordinary Least Squares (OLS) regression.
	"""
	# initialize class
	def __init__(self, fit_intercept=True):
		self.fit_intercept = fit_intercept
	# fit the model
	def fit(self, X, y):
		# if fit_intercept == True
		if (self.fit_intercept) and ('intercept' not in list(X.columns)):
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

# ols k-fold cross validation
def ols_kfold_valid(X, y, int_random_state=42, int_k_folds=10, flt_test_size=0.33, bool_fit_intercept=True, str_metric='r2'):
    """
	Takes an X data frame, y array, random state value, k-folds value, test size proportion, fit intercept boolean, and a metric 
	and completes k-fold OLS regression using train/test split with an average of the provided metric.
    """
    list_eval_metric = []
    for i in range(int_k_folds):
        # add 1 to random_state
        int_random_state += 3
        # split X, y into training, testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=flt_test_size, 
                                                            random_state=int_random_state)
        # instantiate model
        model = OLSRegression(fit_intercept=bool_fit_intercept)
        # fit to training
        model.fit(X=X_train, y=y_train)
        # predict on testing
        model.predict(X_test)
        # evaluate on testing
        dict_eval_metrics = model.evaluate(y=y_test)
        # get the metric
        eval_metric = dict_eval_metrics.get(str_metric)
        # append to list
        list_eval_metric.append(eval_metric)
    # calculate mean r2
    mean_eval_metric = np.mean(list_eval_metric)
    # return mean_eval_metric
    return mean_eval_metric

# define function for fitting catboost model
def fit_catboost_model(X_train, y_train, X_valid, y_valid, list_non_numeric, int_iterations, str_eval_metric, int_early_stopping_rounds, str_task_type='GPU', bool_classifier=True, list_class_weights=None, dict_monotone_constraints=None):
	"""
	Fits a Catboost model for classification or regression.
	"""
	# pool data sets
	# train
	train_pool = cb.Pool(X_train, 
	                     y_train,
	                     cat_features=list_non_numeric)
	# valid
	valid_pool = cb.Pool(X_valid,
	                     y_valid,
	                     cat_features=list_non_numeric) 
	# if fitting classifier
	if bool_classifier:
		# instantiate CatBoostClassifier model
		model = cb.CatBoostClassifier(iterations=int_iterations,
		                              eval_metric=str_eval_metric,
		                              task_type=str_task_type,
		                              class_weights=list_class_weights,
		                              monotone_constraints=dict_monotone_constraints)
	else:
		# instantiate CatBoostRegressor model
		model = cb.CatBoostRegressor(iterations=int_iterations,
		                             eval_metric=str_eval_metric,
		                             task_type=str_task_type,
		                             monotone_constraints=dict_monotone_constraints)
	# fit to training
	model.fit(train_pool,
	          eval_set=[valid_pool], # can only handle one eval set when using gpu
	          use_best_model=True,
	          early_stopping_rounds=int_early_stopping_rounds)
	# return model
	return model