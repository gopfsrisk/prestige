import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import math

# create a binaritizer
class Binaritizer(BaseEstimator, TransformerMixin):
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

# create classifier
class Categorizer(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, dv, list_cols, n_groups, metric='sum', inplace=False, ascending=True):
		self.dv = dv
		self.list_cols = list_cols
		self.n_groups = n_groups
		self.metric = metric
		self.inplace = inplace
		self.ascending = ascending
	# fit
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		list_dict_cat = []
		for col in self.list_cols:
			# group df
			X_grouped = X.groupby(col, as_index=False).agg({self.dv: self.metric})
			# sort df_grouped
			X_grouped = X_grouped.sort_values(by=self.dv, ascending=self.ascending)
			# divide df_grouped into groups
			n_rows_per_group = math.ceil(X_grouped.shape[0]/self.n_groups)
			# mark each row with a group
			counter = 0
			list_group = []
			for i, dv_ in enumerate(X_grouped[self.dv]):
				if i % n_rows_per_group == 0:
					counter += 1
				list_group.append(counter)
			# create keys
			dict_cat = dict(zip(list(X_grouped[col]), list_group))
			list_dict_cat.append(dict_cat)
		# save list_dict_cat to self
		self.list_dict_cat = list_dict_cat
		return self
	# transform
	def transform(self, X):
		for i, col in enumerate(self.list_col):
			# get col name
			if self.inplace:
				new_col = col
			else:
				new_col = '{}_cat'.format(col)
			# map
			X[new_col] = X[col].map(self.list_dict_cat[i])
		return X

# data cleaning function
def cleanme(list_transformers, X_train, y_train, X_test, y_test):
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

# convert to NaN
class ConvertToNaN(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, list_to_NaN, inplace=True):
		self.list_cols = list_cols
		self.list_to_NaN = list_to_NaN
		self.inplace = inplace
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		return self
	# transform
	def transform(self, X):
		for col in self.list_cols:
			if self.inplace:
				X[col] = X[col].replace(self.list_to_NaN, np.nan)
			else:
				X['{0}_to_nan'.format(col)] = X[col].replace(self.list_to_NaN, np.nan)
		return X

# create dummytizer
class Dummytizer(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, drop_first=True, drop_original=True):
		self.list_cols = list_cols
		self.drop_first = drop_first
		self.drop_original = drop_original
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		list_list_val = []
		for col in self.list_cols:
			# get value counts index
			list_val = list(pd.value_counts(X[col]).index)
			# if we want to drop first (reference)
			if self.drop_first:
				# drop first one
				list_val = list_val[1:]
			# append to list
			list_list_val.append(list_val)
		# zip into dictionary
		dict_ = dict(zip(self.list_cols, list_list_val))
		# save to self
		self.dict_ = dict_
		return self
	# transform
	def transform(self, X):
		for col, list_val in self.dict_.items():
			for val in list_val:
				X['{0}_{1}'.format(col, val)] = X.apply(lambda x: 1 if x[col] == val else 0, axis=1)
			# if we want to drop original
			if self.drop_original:
				# drop original col
				X.drop(col, axis=1, inplace=True)
		# return X
		return X

# create mode imputer
class ImputerMode(BaseEstimator, TransformerMixin):
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
	# initialize class
	def __init__(self, list_cols, metric='median', inplace=True):
		self.list_cols = list_cols
		self.metric = metric
		self.inplace = inplace
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# get metric for columns in list_cols
		list_metric_ = []
		for col in self.list_cols:
			# calculate metric
			if self.metric == 'median':
				# calculate median
				metric_ = np.median(X[col].dropna())
			else:
				# calculate mean
				metric_ = np.mean(X[col].dropna())
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

# numeric categorizer
class NumericCategories(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, threshold_len, inplace=True):
		self.list_cols = list_cols
		self.threshold_len = threshold_len
		self.inplace = inplace
	# fit to X
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# get cols with fewer counts than threshold_len
		list_cols_low_len = []
		for col in self.list_cols:
			# get len of value counts
			len_val_counts = len(pd.value_counts(X[col]))
			# logic
			if len_val_counts <= self.threshold_len:
				# append to list
				list_cols_low_len.append(col)
		# save list_cols_low_len to self
		self.list_cols_low_len = list_cols_low_len
		# return self
		return self
	# transform
	def transform(self, X):
		for col in self.list_cols_low_len:
			if self.inplace:
				X[col] = X[col].astype('str')
			else:
				X['{0}_num_cat'.format(col)] = X[col].astype('str')
		# return X
		return X

# remove no var cols
class RemoveNoVar(BaseEstimator, TransformerMixin):
	# intialize class
	def __init__(self):
		pass
	# fit
	def fit(self, X, y=None):
		list_same_col = []
		for col in X.columns:
			if len(pd.value_counts(X[col]).dropna()) <= 1:
				list_same_col.append(col)
		# save to self
		self.list_same_col = list_same_col
		return self
	# transform
	def transform(self, X):
		for col in self.list_same_col:
			if col in list(X.columns):
				X.drop(col, axis=1, inplace=True)
		return X

# create splitter
class SplitterMetric(BaseEstimator, TransformerMixin):
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

# convert cols to string
class StringConverter(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, inplace=True):
		self.list_cols = list_cols
		self.inplace = inplace
	# fit
	def fit(self, X, y=None):
		# make sure all cols in list_cols are in X
		self.list_cols = [col for col in self.list_cols if col in list(X.columns)]
		return self
	# transform
	def transform(self, X):
		for col in self.list_cols:
			if col in list(X.columns):
				if self.inplace:
					X[col] = X[col].astype(str)
				else:
					X['{0}_str'.format(col)] = X[col].astype(str)
		return X

# subset features
class SubsetFeats(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		# make sure all the feats we want are in X
		self.list_mod_feats = [x for x in self.list_cols if x in list(X.columns)]
		return self
	# transform
	def transform(self, X):
		# subset to list_mod_feats
		X = X[self.list_mod_feats]
		return X

# create a target encoder
class TargetEncoder(BaseEstimator, TransformerMixin):
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