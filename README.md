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


<ul>
  <li><a class="active" href="README.md">Home</a></li>
  <li><a href="doc/db_connection.md">Database Connection</a></li>
  <li><a href="doc/preprocessing.md">Preprocessing</a></li>
  <li><a href="doc/model_eval.md">Model Evaluation</a></li>
  <li><a href="doc/general.md">General</a></li>
</ul>

<p align="center"><img src="img/prestige_logo.png" alt="Prestige logo" width=50% height=50% /></p>

---
<h1>prestige (version = 2.0)</h1>

To install:
- (windows) - ```pip install git+https://github.com/gopfsrisk/prestige.git```
- (linux) - ```pip3 install git+https://github.com/gopfsrisk/prestige.git```

To upgrade:
- (windows) - ```pip install git+https://github.com/gopfsrisk/prestige.git -U```
- (linux) - ```pip3 install git+https://github.com/gopfsrisk/prestige.git -U```

---
## Functions for [database connection](doc/db_connection.md), [preprocessing](doc/preprocessing.md), [model evaluation](doc/model_eval.md), and [general purposes](doc/general.md).

---
