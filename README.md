<p align="center"><img src="prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

---
<h1>prestige (version = 2.0)</h1>

To install, use: ```pip install git+https://github.com/aaronengland/prestige.git```

---
<h2>Functions for <a href="#preprocessing">preprocessing</a> and <a href="#general">general</a> purposes.</h2>

---
<a name="preprocessing"></a><h2>prestige.preprocessing</h2>

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

<h3>prestige.preprocessing.cleanme</h3>

<p><i>function</i> prestige.preprocessing.cleanme(<i>list_transformers, X_train, y_train, X_test, y_test</i>)</p>

<p>This function applies each transformer from a list of transformers on train and test data for streamlined data preprocessing.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>list_transformers: <i>list, default=None</i><BR>
		    X_train: <i>dataframe, default=None</i><BR>
		    y_train: <i>series, default=None</i><BR>
		    X_test: <i>dataframe, default=None</i><BR>
		    y_test: <i>series, default=None</i><BR>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>X_train: <i>dataframe, transformed training features</i><BR>
			y_train: <i>series, training dependent variable</i><BR>
			X_test: <i>dataframe, transformed test features</i><BR>
			y_test: <i>series, testing dependent variable</i><BR>
</table>

<p><b>Example:</b></p>

```
import prestige.preprocessing as pre
from sklearn.metrics import MinMaxScaler

# make list_transformers
list_transformers = [pre.Binaritizer(threshold_na=0.1), # create binary features
                     pre.ImputerNumeric(list_cols=list_numeric_cols, metric=metric), # impute medians for numeric columns
                     pre.ImputerMode(list_cols=list_non_numeric_cols), # impute modes for non numeric columns
                     pre.RemoveNoVar(), # remove features with no variance
                     pre.TargetEncoder(list_cols=list_non_numeric_cols, metric='mean'), # convert categorical to numeric
                     pre.ImputerNumeric(list_cols=list_non_numeric_cols, metric=metric) # impute median
                     MinMaxScaler()] # scale data

# apply function
X_train, y_train, X_valid, y_valid = pre.cleanme(list_transformers=list_transformers, 
                                                 X_train=X_train, 
                                                 y_train=y_train, 
                                                 X_test=X_valid, 
                                                 y_test=y_valid)
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

<p><b>Example:</b></p>

```
>>> from prestige.preprocessing import ImputerMode

>>> transformer = ImputerMode(list_cols=list_cols, inplace=True)
>>> X_train = transformer.fit_transform(X_train)
>>> X_valid = transformer.transform(X_valid)
```


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

<p><b>Example:</b></p>

```
>>> from prestige.preprocessing import TargetEncoder

>>> transformer = ImputerNumeric(list_cols=list_cols, metric='mean', rank=False, inplace=True)
>>> X_train = transformer.fit_transform(X_train, y_train)
>>> X_valid = transformer.transform(X_valid)
```

---

<a name="general"></a><h2>prestige.general</h2>

<p>Functions for general purposes and making making code more concise.</p>

#

<h3>prestige.general.agg_one_to_many</h3>

<p><i>function</i> prestige.general.agg_one_to_many(<i>df, unique_id, filename</i>)</p>

<p>This function aggregates by a unique identifier, finds 'min', 'max', 'median', 'mean', 'std', 'sum', and count unique for numeric columns and count unique for non-numeric columns.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>dataframe, default=None</i><BR>
			unique_id: <i>str, default=None</i><BR>
			filename: <i>str, default=None</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>df: <i>dataframe, aggregated dataframe.</i>
</table>


<p><b>Example:</b></p>

```
>>> from prestige.general import agg_one_to_many

>>> df_agg = agg_one_to_many(df=df, unique_id='UniqueID', filename='applications.csv')
```

#

<h3>prestige.general.divide_df</h3>

<p><i>function</i> prestige.general.divide_df(<i>df, date_col, thresh_valid_start, thresh_test_start</i>)</p>

<p>This function splits a dataframe into training, validation, and testing dataframes based on dates.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>dataframe, default=None</i><BR>
			date_col: <i>str, default=None</i><BR>
			thresh_valid_start: <i>datetime, default=None</i><BR>
			thresh_test_start: <i>datetime, default=None</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>df_train, df_valid, df_test: <i>tuple, dataframes.</i>
</table>


<p><b>Example:</b></p>

```
>>> from prestige.general import divide_df

>>> df_train, df_valid, df_test = divide_df(df=df, date_col='application_date', thresh_valid_start=thresh_valid_start, thresh_test_start=thresh_test_start)
```

