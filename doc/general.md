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
>>> dict_ = {'key': value}

>>> # write dictionary to text file
>>> df_agg = dict_to_text(dict_=dict_, 
                          str_filename='dict_.txt')
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
>>> import datetime
>>> from prestige.general import divide_df

>>> # save datetime for when validation data starts
>>> thresh_valid_start = datetime.datetime(year=2016, month=1, day=1)
>>> # save threshold for when test data starts
>>> thresh_test_start = datetime.datetime(year=2016, month=7, day=1)

>>> # divide df into training, validation, and testing
>>> df_train, df_valid, df_test = divide_df(df=df, 
                                            date_col='application_date', 
                                            thresh_valid_start=thresh_valid_start, 
                                            thresh_test_start=thresh_test_start)
```

