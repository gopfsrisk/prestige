import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import skew, kurtosis, shapiro

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

# check for proportion missing in each col
def plot_na(df, filename, tpl_figsize=(10,15)):
	# get the proportion na
	ser_propna = df.isnull().sum()/df.shape[0]
	# sort it
	sort_ser_propna = ser_propna.sort_values(ascending=True)
	# set up axis for plot
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# set title
	ax.set_title('Proportion NaN by column')
	# plot
	ax.barh(sort_ser_propna.index, sort_ser_propna.values)
	# save fig
	plt.savefig(filename, bbox_inches='tight')
	# return sort_ser_propna and fig
	return sort_ser_propna, fig

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

# create descriptive statistics function
def descriptives(df, list_cols, filename):
	# instantiate empty df
	df_empty = pd.DataFrame(index=['count','prop_na','prop_inf','min','max',
	                               'range','mean','median','st_dev','variance',
	                               'skewness','skewness_interp','kurtosis',
	                               'kurtosis_interp','shapiro_pval','shapiro_interp'])
	# iterate through all cols
	for col in list_cols:
	    # extract array
	    ser_col = df[col]
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
	    # put all output into a list
	    list_descriptives = [count, prop_na, prop_inf, min_, max_, range_, mean_,
	                         median_, st_dev, var_, skewness_, skewness_interp,
	                         kurtosis_, kurtosis_interp, shapiro_pval, shapiro_interp]
	    # put list descriptives as col in df_empty
	    df_empty[col] = list_descriptives
	# save df_empty
	df_empty.to_csv(filename, index=True)
	# return df_empty
	return df_empty

# define function for transformation plot grid
def trans_plot_grid(df, list_cols, list_y, str_filename='./img/plt_trans.png', tpl_figsize=(20,20)):
	# create axis
	fig, ax = plt.subplots(nrows=len(list_cols), ncols=7, figsize=tpl_figsize)
	# fix any plot overlap
	plt.tight_layout()
	# iterate through each col doing 6 transformations for each
	for i, col in enumerate(list_cols):
		# set title
		ax[i,0].set_title(col)
		# no transformation
		ax[i,0].scatter(df[col], list_y)
		# set title
		ax[i,1].set_title(f'{col} Squared')
		# squared
		ax[i,1].scatter(df[col]**2, list_y)
		# set title
		ax[i,2].set_title(f'{col} Cubed')
		# cubed
		ax[i,2].scatter(df[col]**3, list_y)
		# set title
		ax[i,3].set_title(f'{col} Square Root')
		# square root
		ax[i,3].scatter(df[col]**(1/2), list_y)
		# set title
		ax[i,4].set_title(f'{col} Cube Root')
		# cube root
		ax[i,4].scatter(df[col]**(1/3), list_y)
		# set title
		ax[i,5].set_title(f'{col} Log')
		# log
		ax[i,5].scatter(np.log10(df[col]), list_y)
		# set title
		ax[i,6].set_title(f'{col} Natural Log')
		# natural log
		ax[i,6].scatter(np.log(df[col]), list_y)
	# save plot
	fig.savefig(str_filename, bbox_inches='tight')
	# return fig
	return fig