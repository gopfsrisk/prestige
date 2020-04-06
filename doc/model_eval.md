<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

---
<a name="model evaluation"></a><h2>prestige.model_eval</h2>

<p>Tools for creating ROC AUC plot.</p>

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
>>>	                      y_hat=y_hat,
>>>	                      figsize=(10,10))
```