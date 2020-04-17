<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <a href="preprocessing.md">Preprocessing</a> | <b><a href="segmentation.md">Segmentation</a></b> | <a href="model_eval.md">Model Evaluation</a> | <a href="general.md">General</a>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="database connection"></a><h2>prestige.db_connection</h2>

<p>Tools for pulling data from a data base into a data frame.</p>

#

<h3>prestige.db_connection.query_to_df</h3>

<p><i>function</i> prestige.db_connection.query_to_df(<i>str_query</i>)</p>

<p>This function takes a SQL string and returns a data frame.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>str_query: <i>str, default=None</i>
	</tr>
	<tr>
		<td>Attributes:</td>
		<td>df: <i>data frame, data frame with data from SQL query</i>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.db_connection import query_to_df

>>> # write query
>>> str_query = """
                SELECT *
                FROM <insert table name>
                """

>>> # pull data
>>> df = query_to_df(str_query=str_query)
```


