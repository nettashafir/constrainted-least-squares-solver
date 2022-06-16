# constrainted-least-squares-solver
A solver for the problem of least squares with the constraint that the solution should be in the unit simplex.

Given matrix H and vector y, try to solve the problem:

```
  min ||Hx-y|| 
   x
    
  s.t. x>=0 
       sum(x)=1
```
For detailed information about the algorithm, please check the file "Project Summary" in this repo.

# Usage
Import the file "solver.py" into your python file.<br>
Use the function "solve", giving it a matrix H and a vector y:
```python
  from solver import solve
  
  if __name__ == "__main__":
    H = ... 
    y = ...
    sol = solve(H, y)
```
