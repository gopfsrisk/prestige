<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <a href="data_exploration.md">Data Exploration</a> | <a href="preprocessing.md">Preprocessing</a> | <a href="segmentation.md">Segmentation</a> | <a href="model_eval.md">Model Evaluation</a> | <b><a href="general.md">General</a></b> | <a href="algorithms.md">Algorithms</a>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="general"></a><h2>prestige.general</h2>

<p>Functions for general purposes and making making code more concise.</p>

#

<h3>prestige.general.dict_to_text</h3>

<p><i>function</i> prestige.general.dict_to_text(<i>dict_, str_filename</i>)</p>

<p>This function writes a dictionary to a text file.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>dict_: <i>dictionary, default=None</i><BR>
			str_filename: <i>str, default=None</i>
	</tr>
</table>


<p><b>Example:</b></p>

```
>>> from prestige.general import dict_to_text

>>> # create dictionary
>>> dict_ = {'key': 'value'}

>>> # write dictionary to text file
>>> dict_to_text(dict_=dict_, 
                 str_filename='dict_.txt')
```

#

<h3>prestige.general.get_numeric_and_nonnumeric</h3>

<p><i>function</i> prestige.general.get_numeric_and_nonnumeric(<i>df, list_ignore_cols</i>)</p>

<p>This function takes a data frame and finds the numeric and non-numeric columns while ignoring provided columns.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>dataframe, default=None</i><BR>
			list_ignore_cols: <i>list, default=None</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>list_numeric, list_non_numeric: <i>tuple, lists.</i>
</table>


<p><b>Example:</b></p>

```
>>> from prestige.general import get_numeric_and_nonnumeric

>>> # get numeric and non-numeric columns
>>> list_numeric, list_non_numeric = get_numeric_and_nonnumeric(df=df,
	                                                        list_ignore_cols=['col1','col2'])
```

