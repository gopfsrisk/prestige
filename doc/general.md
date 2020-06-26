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

#

<h3>prestige.general.list_to_text</h3>

<p><i>function</i> prestige.general.list_to_text(<i>list_items, str_filename, int_rowlength</i>)</p>

<p>This function writes a list to a text file with line breaks at provided row lengths.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>list_items: <i>list, default=None</i><BR>
			str_filename: <i>str, default=None</i><BR>
			int_rowlength: <i>int, defualt=10</i>
	</tr>
</table>


<p><b>Example:</b></p>

```
>>> from prestige.general import list_to_text

>>> # create list
>>> list_ = [1, 2, 3, 4]

>>> # write list to text file
>>> list_to_text(list_items=list_, 
                 str_filename='list_.txt',
                 int_rowlength=2)
```

#

<h3>prestige.general.get_no_var_cols</h3>

<p><i>function</i> prestige.general.get_no_var_cols(<i>df, bool_drop</i>)</p>

<p>This function takes a data frame and finds the columns with no variance.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>dataframe, default=None</i><BR>
			bool_drop: <i>bool, default=True</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>list_no_var_cols, df: <i>tuple, list and dataframe.</i>
</table>


<p><b>Example:</b></p>

```
>>> from prestige.general import get_no_var_cols

>>> # get numeric and non-numeric columns
>>> list_no_var_cols, df = get_no_var_cols(df=df,
	                                   bool_drop=True)
```