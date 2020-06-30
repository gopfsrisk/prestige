import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

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