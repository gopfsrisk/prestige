<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <b><a href="data_exploration.md">Data Exploration</a></b> | <a href="preprocessing.md">Preprocessing</a> | <a href="segmentation.md">Segmentation</a> | <a href="model_eval.md">Model Evaluation</a> | <a href="general.md">General</a>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="data exploration"></a><h2>prestige.data_exploration</h2>

<p>Tools for exploring data.</p>

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

<h3>prestige.data_exploration.plot_na</h3>

<p><i>function</i> prestige.data_exploration.plot_na(<i>df, tpl_figsize, filename</i>)</p>

<p>This function takes a data frame, a tuple indicating figure size, and a string filename (for saving the figure) and saves/returns a bar plot showing the proportion of missing data by column as well as a series with proprtion of missing data by column.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df: <i>df, default=None</i><BR>
			tpl_figsize: <i>tuple, default=(10,15)</i><BR>
			filename: <i>str, defaualt=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.data_exploration import plot_na

>>> # get sorted series of proprtion NaN and generate/save plot
>>> sort_ser_propna, fig_na = plot_na(df=df,
                                      tpl_figsize=(10,15),
                                      filename='./img/plt_prop_na.png')
```

#











