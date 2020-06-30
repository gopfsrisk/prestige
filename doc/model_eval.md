<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <a href="data_exploration.md">Data Exploration</a> | <a href="preprocessing.md">Preprocessing</a> | <a href="segmentation.md">Segmentation</a> | <b><a href="model_eval.md">Model Evaluation</a></b> | <a href="general.md">General</a> | <a href="algorithms.md">Algorithms</a>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="model evaluation"></a><h2>prestige.model_eval</h2>

<p>Tools for evaluating model performance.</p>

#

<h3>prestige.model_eval.roc_auc_curve</h3>

<p><i>function</i> prestige.model_eval.roc_auc_curve(<i>y_true, y_hat, figsize</i>)</p>

<p>This function takes an array of true binary values, an array of predicted probability, and figure size and returns an ROC AUC curve.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>y_true: <i>arr, default=None</i><BR>
			y_hat: <i>arr, default=None</i><BR>
			figsize: <i>tpl, default=(10,10)</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>fig: <i>plot, ROC AUC plot</i>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.model_eval import roc_auc_curve

>>> # generate plot
>>> roc_plot = roc_auc_curve(y_true=y_true,
	                     y_hat=y_hat,
	                     figsize=(10,10))
```

#

<h3>prestige.model_eval.bin_class_eval_metrics</h3>

<p><i>function</i> prestige.model_eval.bin_class_eval_metrics(<i>model_classifier, X, y</i>)</p>

<p>This function takes a classifier model, X, and y. It generates predicted probabilities and predicted classes. It computes accuracy, geometric mean, precision, recall, f1, ROC AUC, precision-recall AUC, log loss, and brier score. The metrics are returned as a dictionary.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>model_classifier: <i>model, default=None</i><BR>
			X: <i>df, default=None</i><BR>
			y: <i>arr, default=None</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>dict_: <i>dict, evaluation metrics</i>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.model_eval import bin_class_eval_metrics

>>> # generate metrics
>>> dict_metrics = bin_class_eval_metrics(model_classifier=model,
	                                  X=X_valid,
	                                  y=y_valid
```