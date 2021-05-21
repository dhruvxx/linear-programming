[![forthebadge](https://forthebadge.com/images/badges/it-works-why.svg)](https://forthebadge.com)

This project uses the **cvxpy** library to solve the optimization problem. \
You can [read the official documentation here](https://www.cvxpy.org/).

#### Contributors
[Pahulpreet Singh](https://github.com/codelixir) and [Dhruvee Birla](https://github.com/dhruvxx).

This assignment was done as a part of the Machine, Data and Learning course, Spring 2021.

---

# Linear Programming

The goal of this exercise is to understand Markov Decision Processes using Mathematical Programming.

Here, we work on the same MDP problem as used in the [value iteration](https://github.com/codelixir/value-iteration/) exercise, with one change:
When MM’s health reaches 0, IJ quest finished but gets zero reward. All other costs and rewards are the same.

In this exercise, we formulate the problem a LP and solve it. The main logic is there in the [notebook](https://github.com/codelixir/linear-programming/blob/main/linearprog.ipynb), and the [script](https://github.com/codelixir/linear-programming/blob/main/code.py) generates the [output](https://github.com/codelixir/linear-programming/blob/main/report/output.json). The approach, observations and conclusions have been summarised in the [report](https://github.com/codelixir/linear-programming/tree/main/report).
