# from http://github.com/stitchfix/diamond
from scipy.special import expit
import numpy as np
from scipy import sparse


def dot(x, y):
    """A generic dot product for sparse and dense vectors
    Args:
        x: array_like, possibly sparse
        y: array_like, possibly sparse
    Returns:
        (dense) dot product of x and y
    """
    if sparse.issparse(x):
        return np.array(x.dot(y))
    return np.dot(x, y)


def l2_clogistic_llh(X, Y, params, alpha):
    """ L2-Penalized log likelihood function for proportional odds cumulative logit model
    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        offset : array_like, optional. Defaults to 0
    Returns:
        scalar : penalized loglikelihood
    """
    J = Y.shape[1]
    intercept, coef = params[:(J-1)], params[(J-1):]
    Xb = dot(X, coef)
    obj = 0
    for j in range(J):
        if j == 0:
            obj += dot(np.log(expit(intercept[j] + Xb)), Y[:, j])
        elif j == J - 1:
            obj += dot(np.log(1 - expit(intercept[j - 1] + Xb)), Y[:, j])
        else:
            obj += dot(np.log(expit(intercept[j] + Xb) -
                       expit(intercept[j - 1] + Xb)), Y[:, j])
    obj -= 0.5 * alpha * np.inner(coef, coef)
    return obj


def l2_clogistic_nllh(X, Y, params, alpha):
    return l2_clogistic_llh(X, Y, params, alpha) * -1


def l2_clogistic_gradient(X, Y, params, alpha):
    """ Gradient of NEGATIVE, penalized LLH of cumulative logit model
    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        params: concatenation of intercepts and coefficients
        alpha: regularization strength
    """
    J = Y.shape[1]
    intercept, coef = params[:(J-1)], params[(J-1):]

    IL = _l2_clogistic_gradient_IL(X=X,
                                   J=J,
                                   params=params)
    grad_coef = _l2_clogistic_gradient_slope(X=X,
                                             Y=Y,
                                             IL=IL,
                                             coef=coef,
                                             alpha=alpha)
    # no regularization of the intercepts
    grad_intercept = _l2_clogistic_gradient_intercept(
            IL=IL,
            Y=Y,
            intercept=intercept)
    return -1.0 * np.concatenate([grad_intercept, grad_coef])


def _l2_clogistic_gradient_IL(X, J, params):
    """ Helper function for calculating the cumulative logistic gradient. \
        The inverse logit of intercept[j + X*coef] is \
        ubiquitous in gradient and Hessian calculations \
        so it's more efficient to calculate it once and \
        pass it around as a parameter than to recompute it every time
    Args:
        X : array_like. design matrix
        intercept : array_like. intercepts. must have shape == one less than the number of columns of `Y`
        coef : array_like. parameters. must have shape == number of columns of X
        offset : array_like, optional. Defaults to 0
        n : int, optional.\
        You must specify the number of rows if there are no main effects
    Returns:
        array_like. n x J-1 matrix where entry i,j is the inverse logit of (intercept[j] + X[i, :] * coef)
    """
    intercept, coef = params[:(J-1)], params[(J-1):]
    n = X.shape[0]
    Xb = dot(X, coef)
    IL = np.zeros((n, J - 1))
    for j in range(J - 1):
        IL[:, j] = expit(intercept[j] + Xb)
    return IL


def _l2_clogistic_gradient_intercept(IL, Y, intercept):
    """ Gradient of penalized loglikelihood with respect to the intercept parameters
    Args:
        IL : array_like. See _l2_clogistic_gradient_IL
        Y : array_like. response matrix
        intercept : array_like. intercepts. must have shape == one less than the number of columns of `Y`
    Returns:
        array_like : length J-1
    """
    exp_int = np.exp(intercept)
    grad_intercept = np.zeros(len(intercept))
    J = Y.shape[1]
    for j in range(J - 1):  # intercepts
        # there are J levels, and J-1 intercepts
        # indexed from 0 to J-2
        if j == 0:
            grad_intercept[j] = dot(Y[:, j], 1 - IL[:, j]) -\
                            dot(Y[:, j + 1], exp_int[j] / (exp_int[j + 1] - exp_int[j]) + IL[:, j])
        elif j < J - 2:
            grad_intercept[j] = dot(Y[:, j], exp_int[j] / (exp_int[j] - exp_int[j - 1]) - IL[:, j]) - \
                            dot(Y[:, j + 1], exp_int[j] / (exp_int[j + 1] - exp_int[j]) + IL[:, j])
        else:  # j == J-2. the last intercept
            grad_intercept[j] = dot(Y[:, j], exp_int[j] / (exp_int[j] - exp_int[j - 1]) - IL[:, j]) - \
                            dot(Y[:, j + 1], IL[:, j])
    return grad_intercept


def _l2_clogistic_gradient_slope(X, Y, IL, coef, alpha):
    """ Gradient of penalized loglikelihood with respect to the slope parameters
    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        IL : array_like. See _l2_clogistic_gradient_IL
        coef : array_like. parameters. must have shape == number of columns of X
        alpha: regularization
    Returns:
        array_like : same length as `coef`.
    """
    grad_coef = np.zeros(len(coef))
    J = Y.shape[1]
    XT = X.transpose()  # CSC format
    for j in range(J):  # coefficients
        if j == 0:
            grad_coef = dot(XT, Y[:, j] * (1.0 - IL[:, j]))
        elif j < J - 1:
            grad_coef += dot(XT, Y[:, j] * (1.0 - IL[:, j] - IL[:, j - 1]))
        else:  # j == J-1. this is the highest level of response
            grad_coef -= dot(XT, Y[:, j] * IL[:, j - 1])
    grad_coef -= alpha * coef
    return grad_coef
