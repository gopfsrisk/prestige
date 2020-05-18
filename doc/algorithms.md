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
		<td>fit_intercept: <i>bool, default=True</i></BR>
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

#

<h3>prestige.segmentation.ols_kfold_valid</h3>

<p><i>function</i> prestige.algorithms.ols_kfold_valid(<i>X, y, int_random_state, int_k_folds, flt_test_size, bool_fit_intercept, str_metric</i>)</p>

<p>This class takes an X data frame, y array, random state value, k-folds value, test size proportion, fit intercept boolean, and a metric and completes k-fold OLS regression using train/test split with an average of the provided metric.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>X: <i>df, default=None</i></BR>
			y: <i>arr, default=None</i></BR>
			int_random_state: <i>int, default=42</i></BR>
			int_k_folds: <i>int, default=10</i></BR>
			flt_test_size: <i>flt, default=0.33</i></BR>
			bool_fit_intercept: <i>bool, default=True</i></BR>
			str_metric: <i>str, default='r2'</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.algorithms import ols_kfold_valid

>>> # find mean r-squared using 10 fold cross-validation
>>> mean_eval_metric = ols_kfold_valid(X=X,
                                       y=y,
                                       int_random_state=42,
                                       int_k_folds=10,
                                       flt_test_size=0.33,
                                       bool_fit_intercept=True,
                                       str_metric='r2')
```




