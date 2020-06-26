import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import os
from sklearn.metrics import roc_auc_score
from itertools import combinations
import random
import catboost as cb
import matplotlib.pyplot as plt
import math
import json

# write dictionary to text
def dict_to_text(dict_, str_filename):
    with open(str_filename, 'w') as file:
        file.write(json.dumps(dict_))

# define function to get numeric and non-numeric cols
def get_numeric_and_nonnumeric(df, list_ignore_cols):
    # instantiate empty lists
    list_numeric = []
    list_non_numeric = []
    # iterate through columns
    for col in df.columns:
        # if its numeric
        if (is_numeric_dtype(df[col])) and (col not in list_ignore_cols):
            # append to list_numeric
            list_numeric.append(col)
        elif (is_numeric_dtype(df[col])==False) and (col not in list_ignore_cols):
            # append to list_non_numeric
            list_non_numeric.append(col)
    # return both lists
    return list_numeric, list_non_numeric

# write list to text file
def list_to_text(list_items, str_filename, int_rowlength=10):
    with open(str_filename, 'w') as file_handler:
        # iterate though each item in list
        for i, item in enumerate(list_items):
            # add 1 to i to make things easier
            i += 1
            # if i == 1
            if i == 1:
                # add open bracket
                file_handler.write(f"['{item}', ")
            # if i is divisible by n_cols and we aren't on the last item
            if (i % int_rowlength == 0) and (i < len(list_items)):
                # start a new line
                file_handler.write(f"'{item}',\n")
            # if we are at the end of the list
            elif i == (len(list_items)):
                # write the final item with no comma
                file_handler.write(f"'{item}']")
            # write item with comma and space
            else:
                file_handler.write(f"'{item}', ")

# drop any columns with zero variance (i.e., all values are the same)
def get_no_var_cols(df, bool_drop=True):
	# get columns with no varince
	list_same_col = []
	for col in df.columns:
		if len(pd.value_counts(df[col]).dropna()) <= 1:
			list_same_col.append(col)
    # if bool_drop 
    if bool_drop:
	   # drop these cols
	   df = df.drop(list_same_col, axis=1, inplace=False)
    # return list_same_col
	return list_same_col, df

# define function for feature engineering
def feature_engineering(df, 
						list_all_eng_feats,
						thresh_n_cols, 
						date_col,
                        thresh_valid_start, 
                        thresh_test_start, 
                        DV,  
                        list_non_numeric_cols, 
                        num_iterations, 
                        early_stopping_rounds, 
                        thresh_imp,
                        list_metrics=['quotient','product','difference','sum_']):
	# get n cols in df
	n_cols_orig = df.shape[1]

	# instantiate counter
	counter = 1
	n_cols = df.shape[1] # current n columns
	for i in range(len(list_all_eng_feats)):
		# get col 1 and col 2
		col_1 = list_all_eng_feats[i][0]
		col_2 = list_all_eng_feats[i][1]
		# set random state
		random.seed(a=i) # index will be our random state so we get different numbers
		# draw a random number in range of length list_metrics
		value_index = random.randint(0, len(list_metrics)-1)
		# get the metric at the value index
		metric = list_metrics[value_index]
		# logic to create col
		if metric == 'quotient':
			col_name = '{0}_div_{1}'.format(col_1, col_2)
			df[col_name] = df[col_1] / df[col_2]
		elif metric == 'product':
			col_name = '{0}_x_{1}'.format(col_1, col_2)
			df[col_name] = df[col_1] * df[col_2]
		elif metric == 'difference':
			col_name = '{0}_-_{1}'.format(col_1, col_2)
			df[col_name] = df[col_1] - df[col_2]
		else:
			col_name = '{0}_+_{1}'.format(col_1, col_2)
			df[col_name] = df[col_1] + df[col_2]

		# --------------------------------------------------------------------
		# replace inf and -inf with 0
		df[col_name] = df[col_name].replace(to_replace=[np.inf, -np.inf], value=0)

		# ---------------------------------------------------------------------
		# get number of cols
		n_cols = df.shape[1]
		# get number of cols added
		n_cols_added = n_cols - n_cols_orig
		# print message
		print('Column {0} created. df contains {1} columns ({2} more than original df; {3}/{4} ({5:0.2f}%) combinations).'.format(col_name, n_cols, n_cols_added, i+1, len(list_all_eng_feats), ((i+1)/len(list_all_eng_feats))*100))
		# if n_cols == 2500 we need to thin out the number of engineered cols to prevent memory errors
		if (n_cols == thresh_n_cols) or (i == len(list_all_eng_feats)-1):
			# replace all -inf or inf values with 0
			for col in df.columns:
				df[col].replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)
			# print message
			print('All inf values have been replaced with 0.')
			# divide into training and testing data
			df_train = df[df[date_col] < thresh_valid_start]
			df_test = df[(df[date_col] >= thresh_valid_start) & (df[date_col] < thresh_test_start)]
			# print message
			print('df divided into training and testing')
        
			# ---------------------------------------------------------------------
			# split each df into X and y
			# df_train
			X_train = df_train.drop([DV, date_col], axis=1)
			y_train = df_train[DV]
			del df_train
			# get X_train.columns
			list_features = list(X_train.columns)
			# df_test
			X_test = df_test.drop([DV, date_col], axis=1)
			y_test = df_test[DV]
			del df_test
			# print message
			print('Training and testing split into X and y')
        
			# ---------------------------------------------------------------------
			# pool data sets
			# train
			train_pool = cb.Pool(X_train, 
								 y_train,
								 cat_features=list_non_numeric_cols)
			del X_train
			del y_train

            # test
			test_pool = cb.Pool(X_test,
								y_test,
								cat_features=list_non_numeric_cols)
			# print message
			print('Pooled dfs created')
        
			# ---------------------------------------------------------------------
			# fit catboost model
			# instantiate catboost model
			model = cb.CatBoostClassifier(iterations=num_iterations,
										  eval_metric='AUC',
										  task_type='GPU')
			# fit to training
			model.fit(train_pool,
					  eval_set=[test_pool], # can only handle one eval set when using gpu
					  use_best_model=True,
					  early_stopping_rounds=early_stopping_rounds)
			del train_pool
			del test_pool
			# ---------------------------------------------------------------------
			# pool X_test
			test_pool = cb.Pool(X_test, cat_features=list_non_numeric_cols)
			# get predicted probabilities
			y_hat_prob = model.predict_proba(test_pool)[:,1]
			# get auc
			auc = roc_auc_score(y_true=y_test, y_score=y_hat_prob)
			del X_test
			del y_test
            
			# ---------------------------------------------------------------------
			# get feature importance
			feat_importance = model.feature_importances_
			# convert to df
			df_imp = pd.DataFrame({'feature': list_features,
								   'importance': feat_importance})	
			# sort
			df_imp = df_imp.sort_values(by=['importance'], ascending=False)
			# write to csv for our own records
			df_imp.to_csv('df_imp_{0}_{1:0.4f}.csv'.format(counter, auc), index=False)
			# increase counter by 1 for next iteration
			counter += 1
        
			#----------------------------------------------------------------------
			# let's get rid of any of the engineered cols that are not in the top n in importance
			list_eng_cols_to_drop = []
			for col in list(df_imp['feature'].iloc[thresh_imp:]):
				if ('_div_' in col) or ('_x_' in col) or ('_-_' in col) or ('_+_' in col):
					# append to list
					list_eng_cols_to_drop.append(col)
			# drop from df
			df.drop(list_eng_cols_to_drop, axis=1, inplace=True)
			# print message
			print('{0} engineered features have been dropped'.format(len(list_eng_cols_to_drop)))
            
			# ---------------------------------------------------------------------
			# write df to csv if i == len(list_all_eng_feats) (if loop is over)
			if i == len(list_all_eng_feats)-1:
				# get number of cols in df
				n_cols_df_final = df.shape[1]
				print('There are {0} columns in df.'.format(n_cols_df_final))
    
				# get number of engineered cols and original cols
				n_eng_cols_df_final = 0
				n_original_cols_df_final = 0
				for col in df.columns:
					if '_div_' in col:
						n_eng_cols_df_final += 1
					elif '_x_' in col:
						n_eng_cols_df_final += 1
					elif '-' in col:
						n_eng_cols_df_final += 1
					elif '+' in col:
						n_eng_cols_df_final += 1
					else:
						n_original_cols_df_final += 1
				print('There are {0} engineered features in df.'.format(n_eng_cols_df_final))
				print('There are {0} original features in df.'.format(n_original_cols_df_final))
	# return the df
	return df

# define helper function for getting all combos
def get_all_combos(list_1, list_2, n_rand_combos, random_state):
	list_list_num_col_combo = []
	for col_1 in list_1:
		for col_2 in list_2:
			list_num_col_combo = [col_1, col_2]
			list_list_num_col_combo.append(list_num_col_combo)
	# some logic
	if len(list_list_num_col_combo) > n_rand_combos:
		# get random subset of n_rand_combos
		random.seed(a=random_state)
		list_list_num_col_combo = random.sample(list_list_num_col_combo, n_rand_combos)
	return list_list_num_col_combo

# define helper function for getting descriptions of engineered cols
def get_descriptions(df):
	# get descriptions for cols
	list_descriptions = []
	for i in range(df.shape[0]):
		# iterate through description and col name
		description = df['Description'].iloc[i]
		col_name = df['Column Name'].iloc[i]
		# if col_name is a list
		if isinstance(col_name, list):
			# get the first element in list for Column Name
			element_1 = df['Column Name'].iloc[i][0]
			# get second element in list for Column Name
			element_2 = df['Column Name'].iloc[i][1]
			# get description of first element
			element_1_desc = df[df['feature'] == element_1]['Description'].values[0]
			# get description of second element
			element_2_desc = df[df['feature'] == element_2]['Description'].values[0]
			# make new description
			description = '{0} divided by {1}'.format(element_1_desc, element_2_desc)
		# append to list
		list_descriptions.append(description)
	# return list_descriptions
	return list_descriptions

# define function to get combinations of cols
def get_numeric_combos(list_numeric_cols, n_rand_combos, random_state):
	list_all_eng_feats = list(combinations(list_numeric_cols, 2))
	# some logic
	if len(list_all_eng_feats) > n_rand_combos:
		# get random subset of n_rand_combos
		random.seed(a=random_state)
		list_all_eng_feats = random.sample(list_all_eng_feats, n_rand_combos)
	return list_all_eng_feats

# define function to get list of lists with combinations of features
def minimum_combinations(list_features, min_feats, max_feats):
	# get all combinations of features
	list_list_feat_combos = []
	for i in range(min_feats, len(list_features)):
		# get feat combos
		list_feat_combos = list(combinations(list_features, i))
		# append to list
		list_list_feat_combos.append(list_feat_combos)
    
	# get all feature combos into lists
	list_model_features = []
	for list_feat_combos in list_list_feat_combos:
		for tuple_feat_combos in list_feat_combos:
			if len(tuple_feat_combos) <= max_feats:
				# convert to list and append
				list_model_features.append(list(tuple_feat_combos))
	# return list_model_features
	return list_model_features

# adverse action (catboost)
def adverse_action(df_shaps, list_unique_id, list_y_actual, list_y_hat, n_reasons=5, drop_list=True):
	# function to get top x (5) shap vals
	def shap_decisions(list_columns, list_shap_vals, n_reasons=n_reasons):
		# create dictionary
		dict_ = dict(zip(list_shap_vals, list_columns))
		# sort dictionary
		dict_sorted = sorted(dict_.items(), reverse=True)
		# get n_reasons 
		reasons = dict_sorted[:n_reasons]
		# create list of reasons
		list_reasons = [x[1] for x in reasons]
		# return list_reasons
		return list_reasons
	# get columns for df_shaps
	list_cols = list(df_shaps.columns)
	# apply shap_decisions function
	list_reasons = list(df_shaps.apply(lambda x: shap_decisions(list_columns=list_cols,
                                                                list_shap_vals=list(x[:]),
                                                                n_reasons=n_reasons), axis=1))
    # create df
	df = pd.DataFrame({'UniqueID': list_unique_id,
                       'actual': list_y_actual,
                       'y_hat': list_y_hat,
                       'reasons': list_reasons})
    # create separate cols for adverse action
	for i in range(n_reasons):
		df['action_{0}'.format(i+1)] = df.apply(lambda x: x['reasons'][i], axis=1)
	# if drop_list is True
	if drop_list:
		# drop reasons
		df.drop('reasons', axis=1, inplace=True)
	# return the df
	return df

