import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# function for ROC curves
def roc_auc_curve(y_true, y_hat, tpl_figsize=(10,10)):
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

