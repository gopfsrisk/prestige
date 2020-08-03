import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import math
from sklearn.utils import resample

# create a binaritizer
class Binaritizer(BaseEstimator, TransformerMixin):
	"""
	Finds the proportion missing of features and converts them to binary if the proportion missing is 
	greater than or equal to threshold_na.
	"""
	# initialize class
	def __init__(self, threshold_na, inplace=False):
		self.threshold_na = threshold_na
		self.inplace = inplace
	# fit to X
	def fit(self, X, y=None):
		# get prop na for each col
		prop_na_df = X.isnull().sum()/X.shape[0]
		# put into df
		df_prop_na = pd.DataFrame({'feature': prop_na_df.index,
								   'prop_na': prop_na_df.values})
		# get list of features where prop_na > threshold
		list_col = []
		for i, col in enumerate(df_prop_na['feature']):
			if df_prop_na['prop_na'].iloc[i] > self.threshold_na:
				list_col.append(col)
		# save to class
		self.list_col = list_col
		return self
	# transform X
	def transform(self, X):
		# convert each col to binary (0 = np.nan, 1 = no nan)
		for col in self.list_col:
			if self.inplace:
				X[col] = X[col].apply(lambda x: 1 if not pd.isnull(x) else 0)
			else:
				X['{0}_bin'.format(col)] = X[col].apply(lambda x: 1 if not pd.isnull(x) else 0)
		# return X
		return X

# data cleaning function
def cleanme(list_transformers, X_train, y_train, X_test, y_test):
	"""
	Applies each transformer from a list of transformers on train and test data for streamlined data 
	preprocessing.
	"""
	for i, transformer in enumerate(list_transformers):
		tool = transformer
		# X_train
		X_train = tool.fit_transform(X_train, y_train)
		# X_test
		X_test = tool.transform(X_test)
		# calculate metrics
		n_complete = i + 1
		# total
		n_total = len(list_transformers)
		# % complete
		pct_complete = (n_complete / n_total) * 100
		# print message
		print('{0}/{1} transformations ({2:0.2f}%) complete.'.format(n_complete, n_total, pct_complete))
	return X_train, y_train, X_test, y_test

# create mode imputer
class ImputerMode(BaseEstimator, TransformerMixin):
	"""
	Imputes each feature's mode for missing values.
	"""
	# initialize class
	def __init__(self, list_cols, inplace=True):
		self.list_cols = list_cols
		self.inplace = inplace
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# get the mode for each col
		list_mode = []
		for col in self.list_cols:
			mode_ = pd.value_counts(X[col].dropna()).index[0]
			# append to list
			list_mode.append(mode_)
		# zip into dictionary
		self.dict_mode = dict(zip(self.list_cols, list_mode))
		return self
	# transform X
	def transform(self, X):
		if self.inplace:
			# fill the nas with dict_mode
			X = X.fillna(value=self.dict_mode, inplace=False)
		else:
			for key, val in self.dict_mode.items():
				X['{0}_imp_mode'.format(key)] = X[key].fillna(value=val, inplace=False)
		return X

# create median imputer
class ImputerNumeric(BaseEstimator, TransformerMixin):
	"""
	Imputes each feature's 'median' or 'mean' for missing values.
	"""
	# initialize class
	def __init__(self, list_cols, metric='median', inplace=True, bool_ignore_neg=True):
		self.list_cols = list_cols
		self.metric = metric
		self.inplace = inplace
		self.bool_ignore_neg = bool_ignore_neg
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# get metric for columns in list_cols
		list_metric_ = []
		for col in self.list_cols:
			# get series with nas dropped
			series_ = X[col].dropna()
			# if bool_ignore_neg
			if self.bool_ignore_neg:
				series_ = series_[series_>=0]
			# calculate metric
			if self.metric == 'median':
				# calculate median
				metric_ = np.median(series_)
			else:
				# calculate mean
				metric_ = np.mean(series_)
			# append to list
			list_metric_.append(metric_)
		# zip into dictionary
		self.dict_metric_ = dict(zip(self.list_cols, list_metric_))
		return self
	# transform X
	def transform(self, X):
		if self.inplace:
			# fill the nas with dict_metric_
			X = X.fillna(value=self.dict_metric_, inplace=False)
		else:
			for key, val in self.dict_metric_.items():
				X['{0}_imp_{1}'.format(key, metric)] = X[key].fillna(value=val, inplace=False)
		return X

# string imputer
class ImputerString(BaseEstimator, TransformerMixin):
	"""
	Imputes a user provided string for NaN values.
	"""
	# initialize class
	def __init__(self, list_cols, str_='MISSING', inplace=True):
		self.list_cols = list_cols
		self.str_ = str_
		self.inplace = inplace
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# get the columns in self.list_cols with missing vals
		list_cols_has_na = []
		list_str_ = []
		for col in self.list_cols:
			if X[col].isnull().sum() > 0:
				list_cols_has_na.append(col)
				list_str_.append(self.str_)
		# create dictionary
		self.dict_ = dict(zip(list_cols_has_na, list_str_))
		return self
	# transform
	def transform(self, X):
		if self.inplace:
			# fill the nas with str_
			X = X.fillna(value=self.dict_, inplace=False)
		else:
			for key, val in self.dict_.items():
				X['{0}_imp_str'.format(key)] = X[key].fillna(val, inplace=False)
		return X

# create splitter
class SplitterMetric(BaseEstimator, TransformerMixin):
	"""
	Creates binary column where values >= median or mean are converted to 1 and others are 0.
	"""
	# initialize class
	def __init__(self, list_cols, metric='median', inplace=False):
		self.list_cols = list_cols
		self.metric = metric
		self.inplace = inplace
	# fit
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# get metric for each numeric column
		list_metric_ = []
		for col in self.list_cols:
			if self.metric == 'median':
				metric_ = np.median(X[col])
			elif self.metric == 'mean':
				metric_ = np.mean(X[col])					
			list_metric_.append(metric_)
		# zip into dict
		self.dict_ = dict(zip(self.list_cols, list_metric_))
		return self
	# transform
	def transform(self, X):
		# iterate through dict
		for key, value in self.dict_.items():
			# convert to binary
			if self.inplace:
				new_col = key
			else:
				new_col = '{0}_{1}_split'.format(key, self.metric)
			# convert to binary
			X[new_col] = X.apply(lambda x: 1 if x[key] >= value else 0, axis=1)
		return X

# create a target encoder
class TargetEncoder(BaseEstimator, TransformerMixin):
	"""
	Converts categorical features into numeric by taking the central tendency metric of the outcome by category.
	"""
	# initialize class
	def __init__(self, list_cols, metric='mean', rank=False, inplace=True):
		self.list_cols = list_cols
		self.metric = metric
		self.rank = rank
		self.inplace = inplace
	# fit to X
	def fit(self, X, y):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# put y as a col in X
		X['dv'] = y
		list_dict_ = []
		# iterate through columns
		for col in self.list_cols:
			# group by the col and get mean of dv (y)
			X_grouped = X.groupby(col, as_index=False).agg({'dv': self.metric})
			# logic if rank 
			if self.rank:
				# rank the dv
				X_grouped['rank'] = X_grouped['dv'].rank()
				# zip into dictionary
				dict_ = dict(zip(list(X_grouped[col]), list(X_grouped['rank'])))
			else:
				dict_ = dict(zip(list(X_grouped[col]), list(X_grouped['dv'])))
			# append dictionary to list
			list_dict_.append(dict_)
		# save lists to self
		self.list_dict_ = list_dict_
		# drop dv from X
		X.drop('dv', axis=1, inplace=True)
		return self
	# transform X
	def transform(self, X):
		# iterate through lists and map them
		for i, col in enumerate(self.list_cols):
			if self.inplace:
				X[col] = X[col].map(self.list_dict_[i])
			else:
				X['{0}_targ_enc'.format(col)] = X[col].map(self.list_dict_[i])
		return X

# define function for oversampling
def oversample(X_train, y_train, int_random_state=42):
	# re-create df_train
	df_train = pd.concat([X_train, y_train], axis=1)

	# separate minority and majority classes
	df_majority = df_train[df_train[y_train.name]==0]
	df_minority = df_train[df_train[y_train.name]==1]

	# upsample minority
	df_minority_upsampled = resample(df_minority,
	                                 replace=True, # sample with replacement
	                                 n_samples=len(df_majority), # match number in majority class
	                                 random_state=int_random_state)

	# combine majority and upsampled minority
	df_train = pd.concat([df_majority, df_minority_upsampled])

	# split into X_train and y_train
	X_train = df_train.drop(y_train.name, axis=1, inplace=False)
	y_train = df_train[y_train.name]

	# return
	return X_train, y_train

# define function for undersampling
def undersample(X_train, y_train, int_random_state=42):
	# re-create df_train
	df_train = pd.concat([X_train, y_train], axis=1)

	# separate minority and majority classes
	df_majority = df_train[df_train[y_train.name]==0]
	df_minority = df_train[df_train[y_train.name]==1]

	# undersample majority
	df_majority_undersampled = resample(df_majority,
	                                    replace=True, # sample with replacement
	                                    n_samples=len(df_minority), # match number in minority class
	                                    random_state=int_random_state)

	# combine df_minority and df_majority_undersampled
	df_train = pd.concat([df_minority, df_majority_undersampled])

	# split into X_train and y_train
	X_train = df_train.drop(y_train.name, axis=1, inplace=False)
	y_train = df_train[y_train.name]

	# return
	return X_train, y_train