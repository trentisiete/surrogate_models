# @date: 19/11/2025
# @author: Jose Arbelaez

from typing import Optional, Union, Any
import numpy as np
from src.utils.paths import get_config_path
from src.models.base_model import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel

class GaussianProcessWrapper(BaseModel):
    """Gaussian Process Regression Model Wrapper of scikit-learn's.
    """
    def __init__(self, kernel: Optional[Kernel]=None, **kwargs) -> None:
        """
        Initializes the Gaussian Process Regressor with the given kernel and
        parameters.
        Args:
            kernel: (Kernel | None = None): If None, the kernel ConstantKernel(1.0,
                constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
                is used as default

            *=None,
            alpha: float | Any = 1e-10,
            optimizer: ((...) -> Any) | Literal['fmin_l_bfgs_b'] | None = "fmin_l_bfgs_b",
            n_restarts_optimizer: Int = 0,
            normalize_y: bool = False,
            copy_X_train: bool = True,
            n_targets: Int | None = None,
            random_state: Int | None = None
        """
        if kernel is None:
            # To ensure that hyperparams are not optimized during fitting
            kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed")

        self.gp = GaussianProcessRegressor(kernel=kernel, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessWrapper":
        """Fit the Gaussian Process model with X and y.

        Args:
            X (np.ndarray): Features for training.
            y (np.ndarray): Target values for training.
        Returns:
            GaussianProcessWrapper: The fitted Gaussian Process model.
        """
        # TODO: If in the future we have problems here, it could be occasioned by
        # not normalizing X and y. Then, we would add here a fit_transform(X)

        self.gp.fit(X, y) # This is correct, in sklearn objects are mutables/in-place
        return self

    def predict(self, X: np.ndarray, return_std: bool = False, return_cov: bool = False) -> Union[np.ndarray, tuple]:
        """
        Predict target values of X matrix

        Args:
            X (np.ndarray): Features for prediction.
            return_std (bool, optional): Whether to return uncertainty.
            Defaults to False.
            return_cov (bool, optional): Whether to return covariance matrix.
            Defaults to False.

        Returns:
            y_mean (ArrayLike): Mean of predictive distribution at query points.
            y_std (ArrayLike): Standard deviation of predictive distribution at
            query points.
            y_cov (ArrayLike): Covariance of joint predictive distribution at
            query poins

        """
        # TODO: If in the future we have problems here, it could be occasioned by
        # not normalizing X. Then, we would add here a transform()

        # It's cool using return_cov but it cost is O(n^3) in RL is better not to abuse it
        return self.gp.predict(X, return_std=return_std, return_cov=return_cov)