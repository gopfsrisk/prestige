<p align="center"><img src="prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

---
<h1>prestige (version = 2.0)</h1>

To install, use: ```pip install git+https://github.com/aaronengland/prestige.git```

---
<h2>prestige.preprocessing</h2>

<p>Tools for data cleaning and preparation.</p>

---
<h3>prestige.preprocessing.Binaritizer</h3>

<p><i>class</i> prestige.preprocessing.Binaritizer(<i>threshold_na, inplace=False</i>)</p>

<p>This estimator find the proportion missing of features and converts them to binary if the proportion missing is greater than or equal to <i>threshold_na</i>.</p>

<table>
	<tr>
		<td width=30%>Parameters:</td>
		<td width=70%>threshold_na: <i>float, default=None</i><BR>
		    inplace: <i>boolean, default=True</i>
	</tr>
	<tr>
		<td width=30%>Attributes:</td>
		<td width=70%>list_col: <i>list of features that were transformed</i>
</table>





