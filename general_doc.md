<p align="center"><img src="img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

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

>>> # aggregate df by the categorical column (i.e., 'UniqueID')
>>> df_agg = agg_one_to_many(df=df, 
                             unique_id='UniqueID', 
                             filename='applications.csv')
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

