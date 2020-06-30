<p align="center">
	<a href="../README.md">Home</a> | <a href="db_connection.md">Database Connection</a> | <a href="data_exploration.md">Data Exploration</a> | <a href="preprocessing.md">Preprocessing</a> | <a href="segmentation.md">Segmentation</a> | <a href="model_eval.md">Model Evaluation</a> | <a href="general.md">General</a> | <b><a href="algorithms.md">Algorithms</a></b>
</p>

---

<p align="center"><img src="../img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

#

<a name="segmentation"></a><h2>prestige.algorithms</h2>

<p>Tools for modeling.</p>

#

<h3>prestige.algorithms.OLSRegression</h3>

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

<h3>prestige.algorithms.ols_kfold_valid</h3>

<p><i>function</i> prestige.algorithms.ols_kfold_valid(<i>X, y, int_random_state, int_k_folds, flt_test_size, bool_fit_intercept, str_metric</i>)</p>

<p>This function takes an X data frame, y array, random state value, k-folds value, test size proportion, fit intercept boolean, and a metric and completes k-fold OLS regression using train/test split with an average of the provided metric.</p>

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

#

<h3>prestige.algorithms.fit_catboost_model</h3>

<p><i>function</i> prestige.algorithms.fit_catboost_model(<i>X_train, y_train, X_valid, y_valid, list_non_numeric, int_iterations, str_eval_metric, int_early_stopping_rounds, str_task_type, bool_classifier</i>)</p>

<p>This function wraps CatBoostClassifier and CatBoostRegressor to make fitting these models. It returns the fit model.</p>

<table>
	<tr>
		<td>Parameters:</td>
		<td>X_train: <i>df, default=None</i></BR>
			y_train: <i>arr, default=None</i></BR>
			X_valid: <i>df, default=None</i></BR>
			y_valid: <i>arr, default=None</i></BR>
			list_non_numeric: <i>list, default=None</i><BR>
			int_iterations: <i>int, default=None</i><BR>
			str_eval_metric: <i>str, default=None</i><BR>
			int_early_stopping_rounds: <i>int, default=None</i><BR>
			str_task_type: <i>str, default='GPU'</i><BR>
			bool_classifier: <i>bool, default=True</i>
	</tr>
</table>

<p><b>Example:</b></p>

```
>>> from prestige.algorithms import fit_catboost_model

>>> # fit catboost classifier model
>>> model = fit_catboost_model(X_train=X_train,
	                           y_train=y_train,
	                           X_valid=X_valid,
	                           y_valid=y_valid,
	                           list_non_numeric=list_non_numeric,
	                           int_iterations=1000,
	                           str_eval_metric='BrierScore',
	                           int_early_stopping_rounds=100,
	                           str_task_type='GPU',
	                           bool_classifier=True)
```



