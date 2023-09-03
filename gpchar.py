from typing import Callable

import numpy as np
import sklearn.gaussian_process as sgp
import plotly.graph_objects as go
import scipy.optimize as sciopt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GPChar:
    """
    Class for characterizing a function using a gaussian process regression.
    """

    def __init__(
        self,
        f: Callable,
        bounds: list[tuple],
        n_features: int,
        n_targets: int,
        previous_evaluation_points: np.ndarray[float] = None,
        previous_evaluation_values: np.ndarray[float] = None,
        n_initial_samples: int = 20,
        keyword_args: dict = None,
    ):
        """
        Constructs instance from function

        Args:
            f (Callable): function to be characterized
            bounds (list[tuple]): bounds for the input values as list of (lo, hi)
                tuples
            input_dim_names (list[str]): Names for the input dimensions. Used as axis
                labels
            output_dim_names (list[str]): Names for the output dimensions. Used as axis
                labels
            n_initial_samples (int): if previous evaluation points are provided,
                this has no effect. Otherwise, this many points are randomly sampled
                before a GP is fit to the data.
            previous_evaluation_points (np.ndarray[float]): points where f has
                been evaluated previously. Defaults to None.
            previous_evaluation_values (np.ndarray[float]): corresponding values
                of f. Defaults to None.
            keyword_args (dict): Keyword arguments to be sent to f. Defaults to None.
        """

        self.f = f
        self.bounds = bounds
        self.n_features = n_features
        self.n_targets = n_targets
        self.keyword_args = keyword_args if keyword_args is not None else dict()

        if previous_evaluation_values is None and previous_evaluation_points is None:
            self.evaluation_points = self.get_random_uniform_points(n_initial_samples)
            self.evaluation_values = np.array([f(point, **self.keyword_args) for point in self.evaluation_points])

        elif ((previous_evaluation_values is None and previous_evaluation_points is not None) or
              (previous_evaluation_values is not None and previous_evaluation_points is None)):
            # Error
            pass
        else:
            self.evaluation_points = previous_evaluation_points
            self.evaluation_values = previous_evaluation_values

        # TODO: use matern kernel
        # if i use a noise model as well I need some scale on the f
        length_scale = [(b[1] - b[0])/5 for b in bounds]
        length_scale_bounds = [((b[1]-b[0])/100, (b[1]-b[0])*10) for b in bounds]
        kernel = (
            1.0 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
            #+ WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)
        )


        # Should I normalize y?
        self.gpr = sgp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gpr.fit(self.evaluation_points, self.evaluation_values)
        print(self.evaluation_points)

        # TODO: something to say which variance should be used to decide what point(s)
        # are the most uncertian

    def get_random_uniform_points(self, n: int) -> np.ndarray[float]:
        """
        Finds n random points in bounds
        
        Args:
            n (int): number of points to return
        Returns:
            points (np.ndarray[float]): points in bounds, n x n_features
        """
        points = np.random.rand(n, self.n_features)
        points = points * np.array([b[1] - b[0] for b in self.bounds]) + np.array([b[0] for b in self.bounds])
        return points

    def acquisition_function(self, x):
        prediction, std = self.gpr.predict(x, return_std=True)
        return std

    def acquire_new_evaluation(self):
        """
        finds new point to evaluate function at in order to maximize gained information
        and evaluates function there, adding the data to the evaluation_points and 
        evaluation_values arrays
        """
        # Minimize acquisition function (variance?)
        res = sciopt.minimize(self.acquisition_function, self.get_random_uniform_points(1)[0], bounds=self.bounds)
        self.evaluation_points = np.vstack(self.evaluation_points, res.x)
        self.evaluation_values = np.vstack(self.evaluation_values, f(res.x, **self.keyword_args))
        self.gpr.fit(self.evaluation_points, self.evaluation_values)

    def get_1d_prediction(self, dimension: int, point: np.ndarray[float]) -> np.ndarray[float]:
        """
        predicts f along dimension where other dimensions are given by point

        Args:
            dimension (int): the numerical dimension along which values will be predicted
            point (np.ndarray[float]): 1d, n_features long numpy array that decides the fixed values
                of all dimensions other than dimension. Thus point[dimension] doesn't matter.

        Returns:
            xs (np.ndarray[int]): the values along dimension where values are predicted
            ys (np.ndarray[int]): the predicted values
            stds (np.ndarray[int]): the predicted std of the predicted values
        """
        # Construct the points to be predicted
        xs = np.linspace(*self.bounds[dimension], 100)
        points = np.tile(point, (100,1))
        points[:,dimension] = xs

        ys, stds = self.gpr.predict(points, return_std=True)
        return xs, ys, stds

    def get_2d_prediction(self, dimension1: int, dimension2: int, point: np.ndarray[float]) -> np.ndarray[float]:
        """
        predicts f along two dimesions: dimension1 and dimension2,
        where other dimensions are given by point

        Args:
            dimension1 (int): the first numerical dimension along which values will be predicted
            dimension2 (int): the second numerical dimension along which values will be predicted
            point (np.ndarray[float]): 1d, n_features long numpy array that decides the fixed values
                of all dimensions other than dimension. Thus point[dimension] doesn't matter.

        Returns:
            x1s (np.ndarray[int]): the values along dimension where values are predicted
            x2s (np.ndarray[int]): the values along dimension where values are predicted
            ys (np.ndarray[int]): the predicted values
            stds (np.ndarray[int]): the predicted std of the predicted values
        """
        # Construct the points to be predicted
        n1 = 100
        n2 = 100
        x1s = np.linspace(*self.bounds[dimension1], n1)
        x2s = np.linspace(*self.bounds[dimension2], n2)
        points = np.tile(point, (n1*n2,1))
        points[:,dimension1] = np.repeat(x1s, n2)
        points[:,dimension2] = np.tile(x1s, n1)

        ys, stds = self.gpr.predict(points, return_std=True)
        return x1s, x2s, ys, stds
