<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <a href="data_exploration.md">Data Exploration</a> | <a href="preprocessing.md">Preprocessing</a> | <b><a href="segmentation.md">Segmentation</a></b> | <a href="model_eval.md">Model Evaluation</a> | <a href="general.md">General</a>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="segmentation"></a><h2>prestige.segmentation</h2>

<p>Tools for segmenting data.</p>

#

<h3>prestige.segmentation.plot_inertia</h3>

<p><i>function</i> prestige.db_connection.plot_inertia(<i>df_X, n_max_clusters, tpl_figsize, title_fontsize, axis_fontsize, str_figname</i>)</p>

<p>This function takes a scaled data frame, a number of clusters, and returns a plot to determine the best number of clusters.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>df_X: <i>df, default=None</i></BR>
		    n_max_clusters: <i>int, default=None</i><BR>
		    tpl_figsize: <i>tpl, default=None</i><BR>
		    title_fontsize: <i>int, default=None</i><BR>
		    axis_fontsize: <i>int, default=None</i><BR>
		    str_figname: <i>str, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.segmentation import plot_inertia

>>> # generate/save plot
>>> fig = plot_inertia(df_X=X,
	                   n_max_clusters=20,
	                   tpl_figsize=(15,20),
	                   title_fontsize=20,
	                   axis_fontsize=15,
	                   str_figname='inertia_plot')
```


