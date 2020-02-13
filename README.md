<p align="center"><img src="prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

---
<h1>prestige (version = 2.0)</h1>

To install, use: ```pip install git+https://github.com/aaronengland/prestige.git```

---
<h2>prestige.preprocessing</h2>

<p>Tools for data cleaning and preparation.</p>

#

<h3>prestige.preprocessing.Binaritizer</h3>

<p><i>class</i> prestige.preprocessing.Binaritizer(<i>threshold_na, inplace=False</i>)</p>

<p>This estimator find the proportion missing of features and converts them to binary if the proportion missing is greater than or equal to <i>threshold_na</i>.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>threshold_na: <i>float, default=None</i><BR>
		    inplace: <i>boolean, default=True</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>list_col: <i>list, features that were transformed</i>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.preprocessing import Binaritizer

>>> transformer = Binaritizer(threshold_na=0.5, inplace=False)
>>> X_train = transformer.fit_transform(X_train)
>>> X_valid = transformer.transform(X_valid)
```

#

<h3>prestige.preprocessing.ImputerNumeric</h3>

<p><i>class</i> prestige.preprocessing.ImputerNumeric(<i>list_cols, metric='median', inplace=True</i>)</p>

<p>This estimator imputes each feature's 'median' or 'mean' for missing values.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>list_cols: <i>list, default=None</i><BR>
		    metric: <i>str, default='median'</i><BR>
		    inplace: <i>boolean, default=True</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>dict_metric_: <i>dict, features and corresponding metric</i>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.preprocessing import ImputerNumeric

>>> transformer = ImputerNumeric(list_cols=list_cols, metric='mean', inplace=True)
>>> X_train = transformer.fit_transform(X_train)
>>> X_valid = transformer.transform(X_valid)
```

#

<h3>prestige.preprocessing.ImputerMode</h3>

<p><i>class</i> prestige.preprocessing.ImputerMode(<i>list_cols, inplace=True</i>)</p>

<p>This estimator imputes each feature's mode for missing values.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>list_cols: <i>list, default=None</i><BR>
		    inplace: <i>boolean, default=True</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>dict_mode: <i>dict, features and corresponding mode</i>
</table>

#

<h3>prestige.preprocessing.TargetEncoder</h3>

<p><i>class</i> prestige.preprocessing.TargetEncoder(<i>list_cols, metric='mean', rank=False, inplace=True</i>)</p>

<p>This estimator converts categorical features into numeric by taking the central tendency metric of the outcome by category.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>list_cols: <i>list, default=None</i><BR>
			metric: <i>str, default='mean'</i><BR>
			rank: <i>boolean, default=False</i><BR>
		    inplace: <i>boolean, default=True</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>list_dict_: <i>list, categories within each feature end corresponding central tendency metric of the outcome variable.</i>
</table>

