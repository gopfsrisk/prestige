<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <a href="data_exploration.md">Data Exploration</a> | <a href="preprocessing.md">Preprocessing</a> | <a href="segmentation.md">Segmentation</a> | <a href="model_eval.md">Model Evaluation</a> | <a href="general.md">General</a> | <b><a href="algorithms.md">Algorithms</a></b>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="segmentation"></a><h2>prestige.algorithms</h2>

<p>Tools for modeling.</p>

#

<h3>prestige.segmentation.OLSRegression</h3>

<p><i>class</i> prestige.algorithms.OLSRegression(<i>fit_intercept</i>)</p>

<p>This class takes an X data frame and y array and completes Ordinary Least Squares (OLS) regression.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>fit_intercept: <i>bool, default=True</i>
			X: <i>df, default=None</i></BR>
		    y: <i>arr, default=None</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.algorithms import OLSRegression

>>> # instantiate model
>>> model = OLSRegression(fit_intercept=True)

>>> # fit to training data
>>> model.fit(X=X_train,
	          y=y_train)

>>> # get a dictionary of beta coefficients
>>> dict_col_betas = model.dict_col_betas

>>> # generate predictions on test data
>>> list_predictions = model.predict(X=X_test)

>>> # generate dictionary of model eval metrics
>>> dict_eval_metrics = model.evaluate(y=y_test)
```


