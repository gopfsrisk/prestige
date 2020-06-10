import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import math
from scipy.stats import skew, kurtosis, shapiro, pearsonr

# get shape of df
def get_shape(df):
	n_rows, n_cols = df.shape
	# print message
	print(f'Rows: {n_rows}')
	print(f'Columns: {n_cols}')
	# return n_rows and n_cols
	return n_rows, n_cols

# check for duplicate rows
def n_dup_rows(df):
	# get n rows in df
	nrows_df = df.shape[0]
	# drop the duplicates
	df_nodups = df.drop_duplicates(subset=None,
		                           keep='first',
		                           inplace=False)
	# get n rows in df_nodups
	nrows_df_nodups = df_nodups.shape[0]
	# substract nrows_df_nodups from df_nodups
	n_duprows = nrows_df - nrows_df_nodups
	# print message
	print(f'There are {n_duprows} duplicate rows in df.')
	# return n_duprows
	return n_duprows

# pie chart of proportion NaN values
def plot_na_overall(df, filename, tpl_figsize=(10,15)):
	# get total number missing
	n_missing = np.sum(df.isnull().sum())
	# get total observations
	n_observations = df.shape[0] * df.shape[1]
	# both into a list
	list_values = [n_missing, n_observations]
	# create axis
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title('Pie Chart of Missing Values')
	ax.pie(x=[n_missing, n_observations], 
	       colors=['y', 'c'],
	       explode=(0, 0.1),
	       labels=['Missing', 'Non-Missing'], 
	       autopct='%1.1f%%')
	# save fig
	plt.savefig(filename, bbox_inches='tight')
	# return fig
	return fig

# check for proportion missing in each col
def plot_na_col(df, filename, tpl_figsize=(10,15), flt_thresh_na=0.05):
	# get the proportion na
	ser_propna = df.isnull().sum()/df.shape[0]
	# sort it
	sort_ser_propna = ser_propna.sort_values(ascending=True)
	# subset to threshold
	ser_propna_sub = sort_ser_propna[sort_ser_propna >= flt_thresh_na]
	# set up axis for plot
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# set title
	ax.set_title('Proportion NaN by column')
	# plot
	ax.barh(ser_propna_sub.index, ser_propna_sub.values)
	# save fig
	plt.savefig(filename, bbox_inches='tight')
	# return sort_ser_propna and fig
	return ser_propna_sub, fig

# NaN heatmap
def plot_na_heatmap(df, filename, tpl_figsize=(20,15), title_fontsize=15):
	# set up axis for plot
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# generate title
	ax.set_title('Missing Data (Yellow) by Column', fontsize=title_fontsize)
	# plot
	sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
	# save fig
	plt.savefig(filename, bbox_inches='tight')
	# return fig
	return fig

# frequency plot of data types
def plot_dtypes_freq(df, filename, tpl_figsize=(15,10)):
	# instantiate empty lists
	list_numeric = []
	list_non_numeric = []
	# iterate through each col
	for col in df.columns:
	    # if its numeric
	    if is_numeric_dtype(df[col]):
	        # append to list
	        list_numeric.append(col)
	    # if its non-numeric
	    else:
	        # append to list
	        list_non_numeric.append(col)
	# get n of each list
	n_numeric = len(list_numeric)
	n_non_numeric = len(list_non_numeric)

	# create axis
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# set title
	ax.set_title('Frequency of Numeric and Non-Numeric Features')
	# set x label
	ax.set_xlabel('Data Type')
	# set y label
	ax.set_ylabel('Frequency')
	# generate plot
	ax.bar(x=['Numeric','Non-Numeric'],
	       height=[n_numeric, n_non_numeric])
	# save fig
	plt.savefig(filename, bbox_inches='tight')
	# return fig
	return list_numeric, list_non_numeric, fig

# generate grid of plots
def plot_grid(df, list_cols, int_nrows, int_ncols, filename, tpl_figsize=(20,15), plot_type='boxplot'):
	# create axes
	fig, ax = plt.subplots(nrows=int_nrows, ncols=int_ncols, figsize=tpl_figsize)
	# fix overlap
	plt.tight_layout()
	# iterate through list_cols
	for i, col in enumerate(list_cols):
    	# get row number by dividing i by int_ncols and taking the floor
		row_number = math.floor(i/int_ncols)
		# get col_number
		if i < int_ncols:
			# col_number is i if we are on top row
			col_number = i
		else:
			# get the column number by dividing i by int_ncols and taking the remainder
			col_number = i % int_ncols
		# replace any inf or -inf with nan and then drop na
		data = df[col].replace(to_replace=[np.inf, -np.inf], value=np.nan).dropna()
		# generate plot
		if plot_type == 'boxplot':
			# set title
			ax[row_number, col_number].set_title(col)
			# generate boxplot
			ax[row_number, col_number].boxplot(data) 
		elif plot_type == 'histogram':
			# generate distplot
			sns.distplot(data, kde=False, color="b", ax=ax[row_number, col_number])
		elif plot_type == 'bar':
			# get frequency table
			ser_freqtbl = pd.value_counts(data)
			# convert index values to string so it doesnt auto sort them
			ser_freqtbl.index = ser_freqtbl.index.map(str)
			# set title
			ax[row_number, col_number].set_title(col)
			# generate bar plot
			ax[row_number, col_number].bar(ser_freqtbl.index, ser_freqtbl.values)
	# save figure   
	fig.savefig(filename, bbox_inches='tight')
	# return fig
	return fig

# define a function to get metrics (this function will be used in descriptives function)
def get_metrics(df, series_):
	# extract array
	ser_col = df[series_]
	# get count
	count = len(ser_col)
	# get proportion NaN
	prop_na = np.sum(ser_col.apply(lambda x: 1 if pd.isnull(x) else 0))/count
	# get proportion inf/-inf
	prop_inf = np.sum(ser_col.apply(lambda x: 1 if math.isinf(x) else 0))/count
	# drop all inf and -inf values
	ser_col = ser_col.replace(to_replace=[np.inf, -np.inf], value=np.nan).dropna()
	# get min
	min_ = np.nanmin(ser_col)
	# get max
	max_ = np.nanmax(ser_col)
	# get range
	range_ = max_ - min_
	# get mean
	mean_ = np.nanmean(ser_col)
	# get median
	median_ = np.nanmedian(ser_col)
	# get st_dev
	st_dev = np.nanstd(ser_col)
	# get variance
	var_ = np.nanvar(ser_col)
	# get skewness
	skewness_ = skew(ser_col)
	# get skewness interpretation
	if skewness_ > 1:
		skewness_interp = 'right skewed'
	elif skewness_ < -1:
		skewness_interp = 'left skewed'
	else:
		skewness_interp = 'not skewed'
	# get kurtosis
	kurtosis_ = kurtosis(ser_col)
	# get kurtosis interpretation
	if kurtosis_ > 1:
		kurtosis_interp = 'too peaked'
	elif kurtosis_ < -1:
		kurtosis_interp = 'too flat'
	else:
		kurtosis_interp = 'not too peaked or too flat'
	# shapiro wilk p-val
	shapiro_pval = shapiro(ser_col)[1]
	# shapiro p_val interpretation
	if shapiro_pval < 0.5:
		shapiro_interp = 'Not normal'
	else:
		shapiro_interp = 'Normal'
	# return the metrics in a tuple
	return count, prop_na, prop_inf, min_, max_, range_, mean_, median_, st_dev, var_, skewness_, skewness_interp, kurtosis_, kurtosis_interp, shapiro_pval, shapiro_interp

# create descriptive statistics function
def descriptives(df, list_cols, str_filename):
	# create list of columns
	list_col_names = ['count','prop_na','prop_inf','min','max', 'range','mean','median','st_dev','variance','skewness','skewness_interp','kurtosis','kurtosis_interp','shapiro_pval','shapiro_interp']
	# create empty df
	df_empty = pd.DataFrame(columns=list_col_names)
	# iterate through list_cols
	for i, col in enumerate(list_cols):
		# apply get_metrics function
		tpl_metrics = get_metrics(df=df,
			                      series_=df[col])
		# print message
		print(f'Descriptives calculated for {i+1}/{len(list_cols)}')
		# zip into dictionary
		dict_row = dict(zip(list_col_names, tpl_metrics))
		# append row to df_empty
		df_empty = df_empty.append(dict_row, ignore_index=True)
	# assign list_cols as index
	df_metrics.index = list_cols
	# write to csv
	df_metrics.to_csv(str_filename, index=True)
	# return df_metrics
	return df_metrics

# define function for transformation plot grid
def trans_plot_grid(df, list_cols, list_y, str_filename='./img/plt_trans.png', tpl_figsize=(20,20)):
	# create axis
	fig, ax = plt.subplots(nrows=len(list_cols), ncols=7, figsize=tpl_figsize)
	# fix any plot overlap
	plt.tight_layout()
	# iterate through each col doing 6 transformations for each
	for i, col in enumerate(list_cols):
		# no transformation
		data = df[col]
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,0].set_title(f'{col} (r = {corr:0.3})') # title
		ax[i,0].scatter(data, list_y) # plot
		# squared
		data = df[col]**2
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,1].set_title(f'{col} Squared (r = {corr:0.3})') # title
		ax[i,1].scatter(data, list_y) # plot
		# cubed
		data = df[col]**3
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,2].set_title(f'{col} Cubed (r = {corr:0.3})') # title
		ax[i,2].scatter(data, list_y) # plot
		# square root
		data = df[col]**(1/2)
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,3].set_title(f'{col} Square Root (r = {corr:0.3})') # title
		ax[i,3].scatter(data, list_y) # plot
		# cube root
		data = df[col]**(1/3)
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,4].set_title(f'{col} Cube Root (r = {corr:0.3})') # title
		ax[i,4].scatter(data, list_y) # plot
		# log
		data = np.log10(df[col])
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,5].set_title(f'{col} Log (r = {corr:0.3})') # title
		ax[i,5].scatter(data, list_y) # plot
		# natural log
		data = np.log(df[col])
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		ax[i,6].set_title(f'{col} Natural Log (r = {corr:0.3})') # title
		ax[i,6].scatter(data, list_y) # plot
	# save plot
	fig.savefig(str_filename, bbox_inches='tight')
	# return fig
	return fig

# define functions for correlation matrix
def correlation_matrix(df, list_cols, bool_abs=False):
	# instantiate empty df
	df_empty = pd.DataFrame()
	# iterate through each column twice
	for col_1 in list_cols:
		list_empty = []
		for col_2 in list_cols:
			# check correlation
			if bool_abs:
				corr = np.abs(pearsonr(df[col_1], df[col_2])[0])
			else:
				corr = pearsonr(df[col_1], df[col_2])[0]
			# append to list
			list_empty.append(corr)
		# append list_empty to df_empty
		df_empty[col_1] = list_empty
	# set index
	df_empty.index = list_cols
	# return df_empty
	return df_empty

# define function for getting the best transformation
def df_best_trans(df, list_cols, list_y, str_id_col, threshold_r=0.5):
	# create empty df
	df_empty = pd.DataFrame(columns=['none', 'squared', 'cubed', 'square_root', 'cube_root', 
	                                 'log', 'natural_log', 'max_abs_val','max_abs_key'])
	# get the best transformation for each feature
	for col in list_cols:
		# insantiate empty list
		dict_empty = {}
		# no transformation
		data = df[col] # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'none': corr}) # append to dict
		# squared
		data = df[col]**2 # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'squared': corr}) # append to dict
		# cubed
		data = df[col]**3 # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'cubed': corr}) # append to dict
		# square root
		data = df[col]**(1/2) # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'square_root': corr}) # append to dict
		# cube root
		data = df[col]**(1/3) # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'cube_root': corr}) # append to dict
		# log
		data = np.log10(df[col]) # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'log': corr}) # append to dict
		# natural log
		data = np.log(df[col]) # get data
		data.replace([np.nan, np.inf, -np.inf], 0, inplace=True) # replace nan and inf with 0
		corr = pearsonr(data, list_y)[0] # pearson
		dict_empty.update({'natural_log': corr}) # append to dict
		# find maximum absolute value of values in dict_empty
		max_abs_val = max([np.abs(val) for val in dict_empty.values()])
		# find key of max absolute val
		max_abs_key = max(dict_empty, key=lambda x: abs(dict_empty[x]))
		# update dict_empty
		dict_empty.update({'max_abs_val': max_abs_val})
		dict_empty.update({'max_abs_key': max_abs_key})
		# append dict_empty as row to df_empty
		df_empty = df_empty.append(dict_empty, ignore_index=True)
    
	# set index of df_empty
	df_empty.index = list_cols

	# subset df_empty to those with max r >= threshold_r
	df_empty_sub = df_empty[df_empty['max_abs_val'] >= threshold_r]

	# create a dictionary with col name and max abs corr
	dict_col_corr = dict(zip(df_empty_sub.index, df_empty_sub['max_abs_val']))
	# create a dictionary with col name and best transformation
	dict_col_tran = dict(zip(df_empty_sub.index, df_empty_sub['max_abs_key']))

	# create a new df with the transformed feats
	df_empty = pd.DataFrame()
	# iterate through index of df_empty_sub
	for i, col in enumerate(df_empty_sub.index):
		# get data
		data = df[col]
		# get best transformation
		best_trans = df_empty_sub['max_abs_key'].iloc[i]
		# logic for transformation
		if best_trans == 'none':
			df_empty[f'{col}_none'] = data
		elif best_trans == 'squared':
			data = data**2
			data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
			df_empty[f'{col}_squ'] = data
		elif best_trans == 'cubed':
			data = data**3
			data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
			df_empty[f'{col}_cub'] = data
		elif best_trans == 'square_root':
			data = data**(1/2)
			data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
			df_empty[f'{col}_sqrt'] = data
		elif best_trans == 'cube_root':
			data = data**(1/3)
			data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
			df_empty[f'{col}_cbrt'] = data
		elif best_trans == 'log':
			data = np.log10(data)
			data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
			df_empty[f'{col}_log'] = data
		elif best_trans == 'natural_log':
			data = np.log(data)
			data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
			df_empty[f'{col}_ln'] = data
        
	# set index
	df_empty.index = df[str_id_col]
	# add profitability column
	df_empty['profitability'] = list_y
	# return
	return dict_col_corr, dict_col_tran, df_empty