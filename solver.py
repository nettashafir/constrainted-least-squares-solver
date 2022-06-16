import numpy as np


# --- Hyperparameters
ALPHA_BACKTRACKING = 0.25
BETA_BACKTRACKING = 0.5
STABILITY_EPS = 1e-6

# GD
CONDITION_NUM_THRESHOLD = 2e9
MAX_STEP_GD = 1000
EPSILON_GD_STOP = 7 * 1e-2

# Newton
EPSILON_QUASY_NEWTON = 0.000001
MAX_STEP_NEWTON = 10
EPSILON_NEWTON_STOP = 1e-4


# --- Helper functions
def euclidean_proj_simplex(x):
    """
    Projection to the unit simplex.
    :param x: The vector to project.
    :return: The projection of x onto the unit simplex.
    """
    # check if we are already on the simplex
    if x.sum() == 1 and np.alltrue(x >= 0):
        # best projection: itself!
        return x
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(x)[::-1]
    cssu = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    K = np.nonzero(u * np.arange(1, x.shape[0] + 1) > (cssu - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssu[K] - 1) / (K + 1)
    # compute the projection by thresholding v using theta
    w = np.clip(a=(x - theta), a_min=0, a_max=None)
    return w


def relaxation_solver(hessian, grad_const):
    """
    This function provides the closed form solution for the problem
    with the sum 1 constraint only.
    :param hessian: The hessian of the least square problem - H.T @ H
    :param grad_const: The constant of the gradient of the least square problem - H.T @ y
    :return: weights vector as solution
    """
    c_vec = np.ones(hessian.shape[0])
    upper_block = np.column_stack((hessian, -c_vec))
    lower_block = np.hstack((c_vec.T, 0))
    kkt = np.vstack((upper_block, lower_block))
    kkt_y = np.hstack((grad_const, 1))

    try:
        beta = np.linalg.solve(kkt, kkt_y)
    except np.linalg.LinAlgError:  # handling singular kkt matrix
        eps_mat = np.eye(kkt.shape[0]) * STABILITY_EPS
        kkt = kkt + eps_mat
        beta = np.linalg.solve(kkt, kkt_y)

    return beta[:-1]


def backtracking_line_search_for_GD(f, x_i, grad, alpha=ALPHA_BACKTRACKING, beta=BETA_BACKTRACKING):
    """
    Back-tracking method for gradient descent algorithm
    :param f: the objective function
    :param x_i: the current x
    :param grad: function mapping points to gradients
    :param alpha: a hyper-parameter
    :param beta: a hyper-parameter
    :return: the optimal step size
    """
    t = 1
    norm_grad_f = grad.dot(grad)
    while f(x_i - t * grad) > f(x_i) - alpha * t * norm_grad_f:
        t *= beta
    return t


def backtracking_line_search_for_Newton(f, x_i, lam_sqr, hessian_inv_dot_grad, alpha=ALPHA_BACKTRACKING,
                                        beta=BETA_BACKTRACKING):
    """
    Back-tracking method for gradient descent algorithm
    :param f: the objective function
    :param x_i: the current x
    :param lam_sqr: hessian squared * grad
    :param hessian_inv_dot_grad:
    :param alpha: a hyper-parameter
    :param beta: a hyper-parameter
    :return: the optimal step size
    """
    t = 1
    while f(x_i - t * hessian_inv_dot_grad) > f(x_i) - alpha * t * lam_sqr:
        t *= beta
    return t


def original_objective(H, y):
    """Least squares objective."""
    return lambda x: np.linalg.norm(H.dot(x) - y)


def least_squares(H, y):
    """Least squares objective."""
    return lambda x: 0.5 * np.linalg.norm(H.dot(x) - y) ** 2


def least_squares_gradient(hessian, grad_const):
    """Gradient of least squares objective at x."""
    return lambda x: hessian.dot(x) - grad_const


def least_squares_hessian(H):
    """Hessian of least squares objective."""
    return H.T.dot(H)


def is_valid(x):
    """ For sanity check """
    return np.isclose(np.sum(x), 1.) and np.alltrue(x >= 0)


# --- Solver
def gradient_descent(init: np.ndarray, H: np.ndarray, y: np.ndarray, hessian: np.ndarray, grad_const: np.ndarray,
                     steps: int = MAX_STEP_GD, epsilon: int = EPSILON_GD_STOP):
    """
    :param init: initial weights for newton process
    :param H: the design matrix of the problem
    :param: y: the response vector of the problem
    :param hessian: The hessian of the least square problem - H.T @ H
    :param grad_const: The constant of the gradient of the least square problem - H.T @ y
    :param steps: the maximum steps for the algorithm
    :return: the best solution done by the algorithm
    """
    f = least_squares(H, y)
    original_f = original_objective(H, y)
    grad = least_squares_gradient(hessian, grad_const)

    prev_f_x = 0
    x_i = init
    for i in range(steps):
        f_x_i = original_f(x_i)
        if abs(prev_f_x - f_x_i) < epsilon:
            break
        grad_x_i = grad(x_i)
        t = backtracking_line_search_for_GD(f, x_i, grad_x_i)
        x_i = euclidean_proj_simplex(x_i - t * grad_x_i)
        prev_f_x = f_x_i
    return x_i


def newton_projected(init: np.ndarray, H: np.ndarray, y: np.ndarray, hessian: np.ndarray, grad_const: np.ndarray,
                     steps: int = MAX_STEP_NEWTON,
                     epsilon: int = EPSILON_NEWTON_STOP):
    """
    Newton method solver
    :param init: initial weights for newton process
    :param H: the design matrix of the problem
    :param: y: the response vector of the problem
    :param hessian: The hessian of the least square problem - H.T @ H
    :param grad_const: The constant of the gradient of the least square problem - H.T @ y
    :return: the best solution done by the algorithm
    """
    f = least_squares(H, y)
    original_f = original_objective(H, y)
    grad = least_squares_gradient(hessian, grad_const)

    hessian_sqrt = np.linalg.cholesky(hessian + EPSILON_QUASY_NEWTON * np.eye(hessian.shape[0]))
    prev_f_x = 0
    x_i = init
    xs = [init]
    for i in range(steps):
        f_x_i = original_f(x_i)
        if abs(prev_f_x - f_x_i) < epsilon:
            break
        lam = np.linalg.solve(hessian_sqrt, grad(x_i))
        hessian_inv_dot_grad = np.linalg.solve(hessian_sqrt.T, lam)
        t = backtracking_line_search_for_Newton(f, xs[-1], lam.dot(lam), hessian_inv_dot_grad)
        x_i = euclidean_proj_simplex(xs[-1] - t * hessian_inv_dot_grad)
        xs.append(x_i)
        prev_f_x = f_x_i
    return min(xs, key=f)


def solve(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    # preprocessing: calculate expensive matrix multiplications
    hessian = H.T.dot(H)
    grad_const = H.T.dot(y)

    # Step 1: try the closed solution
    x_0 = relaxation_solver(hessian, grad_const)

    # Checks if closed solution hold positive constraint
    if np.alltrue(x_0 >= 0):
        return x_0

    # Step 2: Gradient Descent Continue
    condition_number = 0
    if H.shape[0] <= 2000:  # For larger matrices the computation becomes very expensive
        eig = np.linalg.eigh(hessian)[0]
        condition_number = eig[-1] / eig[0] if not np.isclose(eig[0], 0) else np.inf

    # if the first condition holds then the condition number is infinite
    if H.shape[0] > H.shape[1] and condition_number <= CONDITION_NUM_THRESHOLD:
        x_0 = gradient_descent(x_0, H, y, hessian, grad_const)

    # Step 3: Newton method
    x_0 = newton_projected(euclidean_proj_simplex(x_0), H, y, hessian, grad_const)
    return x_0