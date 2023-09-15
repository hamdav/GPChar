from typing import Callable
import threading
import pdb


import numpy as np
import sklearn.gaussian_process as sgp
import plotly.graph_objects as go
import scipy.optimize as sciopt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


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
        save_file: str,
        keyword_args: dict = None,
    ):
        """
        Constructs instance from function

        Args:
            f (Callable): function to be characterized
            bounds (list[tuple]): bounds for the input values as list of (lo, hi)
                tuples
            n_features (int): Number of features, i.e. number of dimensions of the input to f
            n_targets (int): Number of targets, i.e. number of dimensions of the output from f
            save_file (str): filename of file to save data in. If there is already data in this
                file, that is used to initialize the model and new data is appended.
            keyword_args (dict): Keyword arguments to be sent to f. Defaults to None.
        """

        # Initialize class variables
        self.f = f
        self.bounds = bounds
        self.n_features = n_features
        self.n_targets = n_targets
        self.keyword_args = keyword_args if keyword_args is not None else dict()
        self.save_file = save_file
        self.evaluation_points = np.empty((0,n_features))
        self.evaluation_values = np.empty((0,n_targets))
        self.lock = threading.Lock()

        # Load data from save file if there is one
        if save_file is not None:
            try:
                previous_evaluations = np.genfromtxt(save_file, delimiter=",")
                if previous_evaluations.size > 0:
                    self.evaluation_points = previous_evaluations[:,:n_features]
                    self.evaluation_values = previous_evaluations[:,n_features:]
            except FileNotFoundError:
                pass


        length_scale = [(b[1] - b[0])/5 for b in bounds]
        length_scale_bounds = [((b[1]-b[0])/100, (b[1]-b[0])*100) for b in bounds]
        kernel = (
            1.0 * Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=1.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-15, 1e1))
        )


        self.gpr_list = [
            sgp.GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
            )
            for _ in range(n_targets)
        ]

        if self.evaluation_points.size > 0:
            for i, gpr in enumerate(self.gpr_list):
                gpr.fit(self.evaluation_points, self.evaluation_values[:,i])

        # TODO: something to say which variance should be used to decide what point(s)
        # are the most uncertian

    def acquire_random_evaluations_in_thread(self, n: int) -> threading.Thread:
        """ Acquires function evaluations in a separate thread at randomly generated
        points within bounds.

        Args:
            n (int): number of randomly generated points to evaluate f at

        Returns:
            thread (threading.Thread): Thread running the evaluations.
        """
        thread = threading.Thread(
            target=self.acquire_random_evaluations,
            args=(n,)
        )
        thread.start()
        return thread

    def acquire_evaluations_in_thread(self, n: int) -> threading.Thread:
        """ Acquires function evaluations in a separate thread at points
        where the GP is most uncertain.

        Args:
            n (int): number of points to evaluate f at

        Returns:
            thread (threading.Thread): Thread running the evaluations.
        """
        thread = threading.Thread(
            target=self.acquire_new_evaluations,
            args=(n,)
        )
        thread.start()
        return thread
    

    def acquire_random_evaluations(self, n: int):
        """ Acquires function evaluations at randomly generated
        points within bounds. Is thread safe.

        Args:
            n (int): number of randomly generated points to evaluate f at
        """

        for point in self.get_random_uniform_points(n):
            new_value = self.f(point, **self.keyword_args)
            self.lock.acquire()
            self.evaluation_points = np.vstack((self.evaluation_points, point))
            self.evaluation_values = np.vstack((self.evaluation_values, new_value))
            with open(self.save_file, "ab") as f:
                np.savetxt(f, [np.concatenate((point,new_value))],
                           delimiter=",")

            for i, gpr in enumerate(self.gpr_list):
                gpr.fit(self.evaluation_points, self.evaluation_values[:,i])

            self.lock.release()

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
        prediction, std = self.gpr_list[0].predict([x], return_std=True)
        return -float(std)

    def acquire_new_evaluations(self, n: int):
        """
        finds new point to evaluate function at in order to maximize gained information
        and evaluates function there, adding the data to the evaluation_points and 
        evaluation_values arrays. Is thread safe.

        Args:
            n (int): Number of evaluations to perform
        """
        for _ in range(n):
            # Minimize acquisition function (variance?)
            self.lock.acquire()
            res = sciopt.minimize(self.acquisition_function, self.get_random_uniform_points(1)[0], bounds=self.bounds)
            self.lock.release()

            # Calculate new value
            new_value = self.f(res.x, **self.keyword_args)

            # Add value and evaluation point to lists
            self.lock.acquire()
            self.evaluation_points = np.vstack((self.evaluation_points, res.x))
            self.evaluation_values = np.vstack((self.evaluation_values, new_value))
            with open(self.save_file, "ab") as f:
                np.savetxt(f, [np.concatenate((res.x,new_value))],
                           delimiter=",")

            for i, gpr in enumerate(self.gpr_list):
                gpr.fit(self.evaluation_points, self.evaluation_values[:,i])
            self.lock.release()

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

        self.lock.acquire()
        ys = []
        stds = []
        for gpr in self.gpr_list:
            y, std = gpr.predict(points, return_std=True)
            ys.append(y)
            stds.append(std)
        ys = np.column_stack(ys)
        stds = np.column_stack(stds)
        self.lock.release()
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
        points[:,dimension1] = np.tile(x1s, n2)
        points[:,dimension2] = np.repeat(x2s, n1)

        self.lock.acquire()
        ys = []
        stds = []
        for gpr in self.gpr_list:
            y, std = gpr.predict(points, return_std=True)
            ys.append(y)
            stds.append(std)
        ys = np.column_stack(ys)
        stds = np.column_stack(stds)
        self.lock.release()
        return x1s, x2s, ys, stds
