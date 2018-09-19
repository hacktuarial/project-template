import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import optimize
import diamond_utils as utils


class RidgeOrdinalRegressor(BaseEstimator, ClassifierMixin):
    """ Proportional odds ordinal logistic regression with L2 regularization

    Parameters
    ----------
    TODO

    Attributes
    ----------
    TODO
    """
    def __init__(self,
                 alpha=1.0,
                 max_iter=None,
                 tol=1e-3,
                 solver=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        J = y.shape[1]
        assert J > 2

        def fun(params):
            return utils.l2_clogistic_nllh(
                    X=X,
                    Y=y,
                    params=params,
                    alpha=self.alpha)

        def jac(params):
            return utils.l2_clogistic_gradient(
                    X=X,
                    Y=y,
                    params=params,
                    alpha=self.alpha)

        initial_params = np.concatenate([
            np.linspace(-1, 1, J-1),
            np.random.normal(size=X.shape[1])])
        res = optimize.minimize(
                fun=fun,
                jac=jac,
                x0=initial_params,
                method='L-BFGS-B',
                tol=self.tol,
                options={
                    'disp': True,
                    # TODO: pass this along properly
                    # 'maxiter': self.max_iter,
                })
        params = res.x
        self.intercept_ = params[:(J-1)]
        self.coef_ = params[(J-1):]
        return self

    def transform(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
