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


