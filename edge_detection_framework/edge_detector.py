import numpy as np


class EdgeDetector:
    def __init__(self, paths, distances, sqr_derivs=True) -> None:
        """`EdgeDetector` A generic class that gets implemented with different edge detection strategies

        Args:
            paths (_type_): _description_
            distances (_type_): _description_
            sqr_derivs (bool, optional): _description_. Defaults to True.
        """
        # extreme (int, optional): _description_. Defaults to 1000.
        self.paths = paths
        self._distances = distances
        self._sqr_derivs = bool(sqr_derivs)
        # self.extreme = extreme

        if sqr_derivs:
            self._op_derivs = self._sqr
        else:
            self._op_derivs = np.abs

    _nan_row = [(np.nan,) * 3]

    @staticmethod
    def _sqr(l):
        return l**2

    @staticmethod
    def _f_shift(x):
        return np.append(x[1:], EdgeDetector._nan_row, axis=0)

    @staticmethod
    def _b_shift(x):
        return np.append(EdgeDetector._nan_row, x[:-1], axis=0)

    @staticmethod
    def _first_deriv(pixels):
        return EdgeDetector._f_shift(pixels) - EdgeDetector._b_shift(pixels)

    def _get_outliers(self, img: np.ndarray) -> list:
        """`_get_outliers`: Protected method that returns the outliers for the image paths

        Args:
            img (np.ndarray): The image to be processed

        Returns:
            list[np.ndarray]: A list of boolean numpy arrays that match the dimensions of each path.
        """
        pass

    def _get_pixels(self, img):
        return [self._first_deriv(img[y, x]) for x, y in self.paths]

    def _get_sqr_pixels(self, img):
        pixels = self._get_pixels(img)
        return [pix**2 if self._sqr_derivs else np.abs(pix) for pix in pixels]

    def _get_entry(self, i, outliers):
        # if not any(outliers):
        #     # if no edge is found, return a large value
        #     # to simulate the edge being very distant
        #     return (None, None, 1000)
        outliers[-1] = True
        return (
            self.paths[i][0, outliers][0],
            self.paths[i][1, outliers][0],
            self._distances[i][outliers][0],
        )

    def get_edges(self, img):
        return [self._get_entry(i, o) for i, o in enumerate(self._get_outliers(img))]


class IQRDetector(EdgeDetector):
    def __init__(self, paths, distances, distance=None, **kwargs) -> None:
        super().__init__(paths, distances, **kwargs)
        self._distance = distance

    def _iqr(self, x, distance):
        """`iqr(x)`: Function that decides if a given pixel is an outlier

        Args:
            x (np.ndarray): 2d array, of shape (line_length, 3), rgb array of pixels

        Returns:
            np.ndarray: array of outliers detected
        """
        q1 = np.quantile(x[1:-1], 0.25)
        q3 = np.quantile(x[1:-1], 0.75)

        iqr_outlier = distance * (q3 - q1)
        return np.where(
            (x <= q1 - iqr_outlier) | (x >= q3 + iqr_outlier), True, False
        ).any(axis=1)

    # overrides parent's outlier detection function
    def _get_outliers(self, img):
        return [self._iqr(pix, self._distance) for pix in self._get_sqr_pixels(img)]


class StdDetector(EdgeDetector):
    def __init__(
        self, paths, distances, distance=None, use_global=None, basis=None, **kwargs
    ) -> None:
        super().__init__(paths, distances, **kwargs)
        if basis not in ["channels", "mean"]:
            raise ValueError("Basis must be either 'channels' or 'mean'")
        self._distance = distance
        self._use_global = use_global
        self._basis = basis

    def _get_outlier_c(self, line, pixels_all):
        p = pixels_all if self._use_global else line[1:-1]
        return (line > p.mean(axis=0) + self._distance * p.std(axis=0)).any(axis=1)

    def _get_outlier(self, line, pixels_all):
        p = pixels_all if self._use_global else line[1:-1]
        return line.mean(axis=1) > p.mean() + self._distance * p.std()

    # overrides parent's outlier detection function
    def _get_outliers(self, img):
        pixels = self._get_pixels(img)
        pixels_all = np.concatenate([line[1:-1] for line in pixels])

        get_outlier = (
            self._get_outlier_c if self._basis == "channels" else self._get_outlier
        )
        return [get_outlier(line, pixels_all) for line in pixels]


class RollStdDetector(EdgeDetector):
    def __init__(
        self,
        paths,
        distances,
        distance: float = None,
        use_global: bool = None,
        roll_window: int = None,
        scale_roll: bool = False,
        combine_lines: str = "mean",
        **kwargs
    ) -> None:
        super().__init__(paths, distances, **kwargs)
        self._distance = distance
        self._roll_window = roll_window
        self._use_global = use_global
        self._scale_roll = scale_roll

        if combine_lines == "mean":
            self._combine = self._mean
        elif combine_lines == "max":
            self._combine = self._max
        else:
            raise ValueError("combine_lines must be either 'mean' or 'max'")

    @staticmethod
    def _mean(l):
        return l.mean(axis=1)

    @staticmethod
    def _max(l):
        return l.max(axis=1)

    @staticmethod
    def _roll(a, window):
        a_n = np.concatenate([[np.NAN] * (window - 1), a])
        return np.stack([a_n[i - window : i] for i in range(window, a_n.shape[0] + 1)])

    def _scale(self, roll):
        s_min = roll[self._roll_window : -1].min()
        s_max = roll[self._roll_window : -1].max()

        return (roll - s_min) / (s_max - s_min + 1e-10)

    def _get_outlier(self, line):
        roll = self._roll(self._combine(line), self._roll_window)

        # If we need to, scale the roll first
        if self._scale_roll:
            roll = self._scale(roll)

        roll = self._op_derivs(roll)

        r_max = roll.max(axis=1)
        mean = r_max[self._roll_window : -1].mean()
        std = r_max[self._roll_window : -1].std()
        return r_max > mean + self._distance * std

    def _get_outlier_global(self, roll, mean, std):
        return roll > mean + self._distance * std

    # overrides parent's outlier detection function
    def _get_outliers(self, img):
        pixels = self._get_pixels(img)
        if self._use_global:
            roll_pixels = [
                self._roll(self._combine(line), self._roll_window) for line in pixels
            ]

            # If we need to, scale the roll first
            if self._scale_roll:
                roll_pixels = [self._scale(roll) for roll in roll_pixels]

            roll_pixels = [self._op_derivs(roll) for roll in roll_pixels]

            roll_pixels = [line.max(axis=1) for line in roll_pixels]
            all_roll_pixels = np.concatenate(
                [roll[self._roll_window : -1] for roll in roll_pixels]
            )
            mean = all_roll_pixels.mean()
            std = all_roll_pixels.std()
            return [self._get_outlier_global(roll, mean, std) for roll in roll_pixels]
        else:
            return [self._get_outlier(line) for line in pixels]
