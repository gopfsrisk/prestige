<!DOCTYPE html>
<html>
<head>
<style>
ul {
  list-style-type: none;
  margin: 0;
  padding: 0;
  overflow: hidden;
  background-color: #333;
}

li {
  float: left;
}

li a {
  display: block;
  color: white;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
}

li a:hover:not(.active) {
  background-color: #111;
}

.active {
  background-color: #4CAF50;
}
</style>
</head>
<body>

<ul>
  <li><a class="active" href="README.md">Home</a></li>
  <li><a href="doc/db_connection.md">Database Connection</a></li>
  <li><a href="doc/preprocessing.md">Preprocessing</a></li>
  <li><a href="doc/model_eval.md">Model Evaluation</a></li>
  <li><a href="doc/general.md">General</a></li>
</ul>

</body>
</html>
