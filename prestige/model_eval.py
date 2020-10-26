import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import (accuracy_score, fowlkes_mallows_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss)
import prestige.general as gen
from scipy.stats import zscore
import numpy as np

# function for ROC curves
def roc_auc_curve(y_true, y_hat, tpl_figsize=(10,10)):
	"""
	Takes an array of true binary values, an array of predicted probability, and figure size and returns an ROC AUC curve.
	"""
	# get roc auc
	auc = roc_auc_score(y_true=y_true,
		                y_score=y_hat)
	# get false positive rate, true positive rate
	fpr, tpr, thresholds = roc_curve(y_true=y_true, 
		                             y_score=y_hat)
	# set up subplots
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# set title
	ax.set_title('ROC Plot - (AUC: {0:0.4f})'.format(auc))
    # set x axis label
	ax.set_xlabel('False Positive Rate (Sensitivity)')
    # set y axis label
	ax.set_ylabel('False Negative Rate (1 - Specificity)')
    # set x lim
	ax.set_xlim([0,1])
    # set y lim
	ax.set_ylim([0,1])
    # create curve
	ax.plot(fpr, tpr, label='Model')
    # plot diagonal red, dotted line
	ax.plot([0,1], [0,1], color='red', linestyle=':', label='Chance')
    # create legend
	ax.legend(loc='lower right')
	# fix overlap
	plt.tight_layout()
	# return fig
	return fig

# define function to get binary eval metrics
def bin_class_eval_metrics(model_classifier, X, y):
	"""
	Computes binary evaluation metrics and returns them as a dictionary.
	"""
	# generate predicted class
	y_hat_class = model_classifier.predict(X)
	# generate predicted probabilities
	y_hat_proba = model_classifier.predict_proba(X)[:,1]
	# metrics
	# accuracy
	accuracy = accuracy_score(y_true=y, y_pred=y_hat_class)
	# geometric mean
	geometric_mean = fowlkes_mallows_score(labels_true=y, labels_pred=y_hat_class)
	# precision
	precision = precision_score(y_true=y, y_pred=y_hat_class)
	# recall
	recall = recall_score(y_true=y, y_pred=y_hat_class)
	# f1
	f1 = f1_score(y_true=y, y_pred=y_hat_class)
	# roc auc
	roc_auc = roc_auc_score(y_true=y, y_score=y_hat_proba)
	# precision recall auc
	pr_auc = average_precision_score(y_true=y, y_score=y_hat_proba)
	# log loss
	log_loss_ = log_loss(y_true=y, y_pred=y_hat_proba)
	# brier 
	brier = brier_score_loss(y_true=y, y_prob=y_hat_proba)
	# put into dictionary
	dict_ = {'accuracy': accuracy,
	         'geometric_mean': geometric_mean,
	         'precision': precision,
	         'recall': recall,
	         'f1': f1,
	         'roc_auc': roc_auc,
	         'pr_auc': pr_auc,
	         'log_loss': log_loss_,
	         'brier': brier}
	# return dict_
	return dict_

# define function for pd plots
def partial_dependence_plots(model, X_train, y_train, bool_prob=True, int_plot_length=100, str_filename='./img/plt_pdp.png'):
	"""
	Takes a model, data frame of features (X_train), actual target values (y_train),
	boolean for probability (bool_prob), number of subplots in each plot (int_plot_length),
	and a string filename (str_filename) for saving plots.
	Returns partial density plots in the specified filepath.
	"""
	# generate predictions from model
	if bool_prob:
	    # generate predicted probabilities
	    y_hat_train = model.predict_proba(X_train)[:,1]
	else:
	    # generate predicted number (regression)
	    y_hat_train = model.predict(X_train)

	# get numeric and non-numeric features
	list_numeric, list_non_numeric = gen.get_numeric_and_nonnumeric(df=X_train,
	                                                                list_ignore_cols=[])

	# create list of lists with each list length of int_plot_length
	list_large = []
	list_small = []
	for a, col in enumerate(list(X_train.columns)):
	    list_small.append(col)
	    # if there are int_plot_length items in list_small or if we are at the end of list_numeric
	    if (len(list_small)==int_plot_length) or ((a+1)==len(X_train.columns)):
	        # add list_small to list_large
	        list_large.append(list_small)
	        # reset list small
	        list_small = []

	# create dataframe
	X_train['predicted'] = y_hat_train
	X_train['actual'] = y_train

	# generate plots
	for a, list_ in enumerate(list_large):
	    # get number of rows
	    n_rows = len(list_)
	    # get figsize
	    tpl_figsize = (20, 5*n_rows)
	    # create axis
	    fig, ax = plt.subplots(nrows=n_rows, figsize=tpl_figsize)
	    # fix overlap
	    plt.tight_layout()
	    
	    # iterate through each item in list_
	    for b, item_ in enumerate(list_):
	        # group df
	        X_train_grouped = X_train.groupby(by=item_, as_index=False).agg({'predicted': 'mean',
	                                                                         'actual': 'mean'})
	        # numeric
	        if item_ in list_numeric:
	            # sort
	            X_train_grouped = X_train_grouped.sort_values(by=item_, ascending=True)
	            # make z score col name
	            str_z_col = '{0}_z'.format(item_)
	            # get z score
	            X_train_grouped[str_z_col] = zscore(X_train_grouped[item_])
	            # subset to only those with z >= 3 and <= -3
	            X_train_grouped = X_train_grouped[(X_train_grouped[str_z_col] < 3) & (X_train_grouped[str_z_col] > -3)]
	            # calculate trendlines
	            # predicted
	            z_pred = np.polyfit(X_train_grouped[item_], X_train_grouped['predicted'], 1)
	            p_pred = np.poly1d(z_pred)
	            # actual
	            z_act = np.polyfit(X_train_grouped[item_], X_train_grouped['actual'], 1)
	            p_act = np.poly1d(z_act)
	            # plot trendline
	            # predicted
	            ax[b].plot(X_train_grouped[item_], p_pred(X_train_grouped[item_]), color='green', label='Trend - Predicted')
	            # actual
	            ax[b].plot(X_train_grouped[item_], p_act(X_train_grouped[item_]), color='orange', label='Trend - Actual')
	        
	        # plot it
	        ax[b].set_title(item_)
	        # predicted
	        ax[b].plot(X_train_grouped[item_], X_train_grouped['predicted'], color='blue', label='Predicted')
	        # actual
	        ax[b].plot(X_train_grouped[item_], X_train_grouped['actual'], color='red', linestyle=':', label='Actual')
	        
	        # legend
	        ax[b].legend(loc='upper right')
	    
	    # create filename
	    str_filename_new = f'{str_filename[:-4]}_{a}{str_filename[-4:]}'
	    
	    # save fig
	    plt.savefig(str_filename_new, bbox_inches='tight')
	    # close plot
	    plt.close()