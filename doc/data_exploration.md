<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <b><a href="data_exploration.md">Data Exploration</a></b> | <a href="preprocessing.md">Preprocessing</a> | <a href="segmentation.md">Segmentation</a> | <a href="model_eval.md">Model Evaluation</a> | <a href="general.md">General</a> | <a href="algorithms.md">Algorithms</a>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="data exploration"></a><h2>prestige.data_exploration</h2>

<p>Tools for exploring data.</p>

#

<h3>prestige.data_exploration.get_shape</h3>

<p><i>function</i> prestige.data_exploration.get_shape(<i>df</i>)</p>

<p>This function takes a data frame and returns the number of rows and number of columns.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import get_shape

>>> # get number of rows and number of columns
>>> n_rows, n_cols = get_shape(df=df)
```

#

<h3>prestige.data_exploration.n_dup_rows</h3>

<p><i>function</i> prestige.data_exploration.n_dup_rows(<i>df</i>)</p>

<p>This function takes a data frame and returns the number of duplicate rows.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import n_dup_rows

>>> # get number of duplicate rows
>>> n_dup_rows = n_dup_rows(df=df)
```

#

<h3>prestige.data_exploration.drop_prop_nan</h3>

<p><i>function</i> prestige.data_exploration.drop_prop_nan(<i>df, flt_thresh_na</i>)</p>

<p>This function takes a data frame and a threshold for proportion of missing rows and returns lists for columns to drop, columns to keep, and a dataframe with the columns dropped.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			flt_thresh_na: <i>float, default=1.0</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import drop_prop_nan

>>> # drop columns with the proportion provided missing
>>> list_propna_drop, list_propna_keep, df = drop_prop_nan(df=df,
	                                                   flt_thresh_na=1.0)
```

#

<h3>prestige.data_exploration.drop_no_variance</h3>

<p><i>function</i> prestige.data_exploration.drop_no_variance(<i>df</i>)</p>

<p>This function takes a data frame and returns a list of columns with no variance and a data frame with those columns dropped.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import drop_no_variance

>>> # drop columns with no variance
>>> list_no_var, df = drop_no_variance(df=df)
```

#

<h3>prestige.data_exploration.plot_na_overall</h3>

<p><i>function</i> prestige.data_exploration.plot_na_overall(<i>df, filename, tpl_figsize</i>)</p>

<p>This function takes a data frame and returns a pie chart of missing and not missing.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			filename: <i>str, default=None</i><BR>
			tpl_figsize: <i>tpl, default=(10,15)</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import plot_na_overall

>>> # create pie chart of missing and non missing values
>>> fig = plot_na_overall(df=df,
	                  filename='fig_nan_pie.png',
	                  tpl_figsize=(10,15))
```

#

<h3>prestige.data_exploration.plot_na_col</h3>

<p><i>function</i> prestige.data_exploration.plot_na_col(<i>df, tpl_figsize, filename</i>)</p>

<p>This function takes a data frame, a tuple indicating figure size, and a string filename (for saving the figure) and saves/returns a bar plot showing the proportion of missing data by column as well as a series with proprtion of missing data by column.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			tpl_figsize: <i>tuple, default=(10,15)</i><BR>
			filename: <i>str, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import plot_na_col

>>> # get sorted series of proprtion NaN and generate/save plot
>>> sort_ser_propna, fig_na = plot_na_col(df=df,
                                          tpl_figsize=(10,15),
                                          filename='./img/plt_prop_na.png')
```

#

<h3>prestige.data_exploration.plot_na_heatmap</h3>

<p><i>function</i> prestige.data_exploration.plot_na_heatmap(<i>df, tpl_figsize, title_fontsize, filename</i>)</p>

<p>This function takes a data frame, a tuple indicating figure size, a title fontsize, and a string filename (for saving the figure) and saves/returns a heatmap showing the occurences (rows) of missing data by column.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			tpl_figsize: <i>tuple, default=(20,15)</i><BR>
			title_fontsize: <i>int, default=15</i><BR>
			filename: <i>str, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import plot_na_heatmap

>>> # get missing value heatmap
>>> fig_na_heatmap = plot_na_heatmap(df=df,
                                     tpl_figsize=(20,15),
                                     title_fontsize=15,
                                     filename='./img/fig_na_heatmap.png')
```

#

<h3>prestige.data_exploration.plot_dtypes_freq</h3>

<p><i>function</i> prestige.data_exploration.plot_dtypes_freq(<i>df, filename, list_ignore_cols, tpl_figsize</i>)</p>

<p>This function takes a data frame and returns a bar plot of numeric and non-numeric columns.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			filename: <i>str, default=None</i><BR>
			list_ignore_cols: <i>list, default=None</i><BR>
			tpl_figsize: <i>tuple, default=(20,15)</i>	
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import plot_dtypes_freq

>>> # plot the numeric and non-numeric columns
>>> fig_dtypes = plot_dtypes_freq(df=df,
	                          filename='fig_dtypes.png'
                                  list_ignore_cols=['col1','col2'],
                                  tpl_figsize=(20,15))
```

#

<h3>prestige.data_exploration.plot_grid</h3>

<p><i>function</i> prestige.data_exploration.plot_grid(<i>df, list_cols, int_nrows, int_ncols, filename, tpl_figsize, plot_type</i>)</p>

<p>This function takes a data frame, a list of columns, an integer number of rows, an integer number of columns, a string filename (for saving the figure), a tuple indicating figure size, and a string plot type ('boxplot', 'histogram', or 'bar') and saves/returns a grid of desired plot type.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			list_cols: <i>list, default=None</i><BR>
			int_nrows: <i>int, default=None</i><BR>
			int_ncols: <i>int, default=None</i><BR>
			filename: <i>str, default=None</i><BR>
			tpl_figsize: <i>tuple, default=(20,15)</i><BR>
			plot_type: <i>string, default='boxplot'</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import plot_grid

>>> # get plot grid
>>> fig_plot_grid = plot_grid(df=df,
                              list_cols=list_cols,
                              int_nrows=6,
                              int_ncols=7,
                              plot_type='boxplot',
                              filename='./img/plt_boxplotgrid.png'
                              tpl_figsize=(20,15))
```

#

<h3>prestige.data_exploration.descriptives</h3>

<p><i>function</i> prestige.data_exploration.descriptives(<i>df, list_cols, filename</i>)</p>

<p>This function takes a data frame, a list of columns, and a string filename (for saving the figure), and saves/returns a dataframe to a .csv file containing descriptive statistics for each column.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			list_cols: <i>list, default=None</i><BR>
			filename: <i>str, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import descriptives

>>> # get descriptives
>>> df_descriptives = descriptives(df=df,
                                   list_cols=list_cols,
                                   filename='./img/plt_boxplotgrid.png')
```

#

<h3>prestige.data_exploration.distribution_analysis</h3>

<p><i>function</i> prestige.data_exploration.distribution_analysis(<i>df, str_datecol, list_numeric_cols, str_filename, int_length</i>)</p>

<p>This function takes a data frame and a list of numeric columns for which to calculate and plot means by month.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			str_datecol: <i>str, default=None</i><BR>
			list_numeric_cols: <i>list, default=None</i><BR>
			str_filename: <i>str, default=None</i><BR>
			int_length: <i>int, default=100</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import distribution_analysis

>>> # conduct distribution analysis
>>> distribution_analysis(df=df,
                          str_datecol='date_col',
                          list_numeric_cols=['col1','col2'],
                          str_filename='plt_dist.png',
                          int_length=100)
```

















