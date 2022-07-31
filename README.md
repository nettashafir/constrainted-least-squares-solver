# constrainted-least-squares-solver
A solver for the problem of least squares with the constraint that the solution should be in the unit simplex.<br>
Given matrix H and vector y, try to solve the problem:
$$\min_x\ \ \ \ \ \left\Vert Hx-y\right\Vert $$
$$s.t.\ \ \ \ x\geq0 $$
$$\ \ \ \ \ \ \ \ \ \ \ \sum_{i=1}^{n}x_i=1$$

For detailed information about the algorithm, please check the file "Project Summary" in this repo.

This project was carried out as part of the course "Convex Optimization and Application",  at the Hebrew University of Jerusalem.

# Usage
Import the file "solver.py" into your python file.<br>
Use the function "solve", giving it a matrix H and a vector y:
```python
# main.py file
from solver import solve

if __name__ == "__main__":
  H = ... 
  y = ...
  sol = solve(H, y)
```
