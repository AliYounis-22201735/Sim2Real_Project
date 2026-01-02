from interpilotable.model.paths import get_paths
from interpilotable.model.edge_detector import IQRDetector, StdDetector, RollStdDetector
from interpilotable.model.correction import correction

import cv2

import atexit
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, TypedDict, Optional, Union


class LineDrawerParams(TypedDict):
    correction_args: Tuple[float, float, int, float, float, int]
    kernel_size: int
    num_lines: int
    skip_lines: int
    edge_strategy: str
    distance: float
    sqr_derivs: Optional[bool]
    use_global: Optional[bool]
    basis: Optional[str]
    roll_window: Optional[float]
    scale_roll: Optional[bool]
    combine_lines: Optional[str]


class SoloLineDrawer:
    def __init__(
        self,
        num_lines: int,
        skip_lines: int,
        kernel_size: int,
        correction_args: tuple,
        edge_strategy: str = None,
        sqr_derivs: bool = True,
        distance: float = 1,
        use_global: bool = False,
        basis: str = "channels",
        roll_window: float = 5,
        scale_roll: bool = False,
        combine_lines: str = "mean",
    ) -> None:
        self.paths, self.distances, self.target_names = get_paths(
            num_lines=num_lines, skip_lines=skip_lines
        )

        # set up the image pre-processing
        self.correction_args = correction_args
        self.kernel_size = kernel_size

        # set up the edge detection
        if edge_strategy == "iqr":
            self.edge_detector = IQRDetector(
                self.paths, self.distances, distance, sqr_derivs=sqr_derivs
            )
        elif edge_strategy == "std":
            self.edge_detector = StdDetector(
                self.paths,
                self.distances,
                distance,
                use_global,
                basis,
                sqr_derivs=sqr_derivs,
            )
        elif edge_strategy == "roll":
            self.edge_detector = RollStdDetector(
                self.paths,
                self.distances,
                distance,
                use_global,
                roll_window,
                scale_roll,
                combine_lines,
                sqr_derivs=sqr_derivs,
            )
        else:
            raise ValueError("Must specify a valid edge strategy")

    def transform_image(self, z: np.ndarray):
        # transform the image first, then do correction
        z = cv2.GaussianBlur(z, (self.kernel_size, self.kernel_size), 0)
        return correction(z, *self.correction_args) if self.correction_args != None else z

    def draw_lines(self, img: np.ndarray) -> List[Tuple[int, int, float]]:
        z = self.transform_image(img)
        return self.edge_detector.get_edges(z)

    def get_edges(self, img_arr: np.ndarray) -> list:
        # get only the first element
        return [x[2] for x in self.draw_lines(img_arr)]


class LineDrawer:
    def __init__(
        self,
        drawer_1_kwargs: LineDrawerParams,
        drawer_2_kwargs: Optional[LineDrawerParams] = None,
    ) -> None:
        self.d1 = SoloLineDrawer(**drawer_1_kwargs)
        self.angle_target_names = self.d1.target_names

        if drawer_2_kwargs is not None:
            self.solo_drawer = False
            self.d2 = SoloLineDrawer(**drawer_2_kwargs)
            self.throttle_target_names = self.d2.target_names
        else:
            self.solo_drawer = True
            self.throttle_target_names = self.angle_target_names

    def draw_lines(self, img: Union[str, np.ndarray]):
        # read the image if the `img` is a file name, otherwise treat as a loaded image
        z = cv2.imread(img) if isinstance(img, str) else img

        result_1 = self.d1.draw_lines(z)

        if self.solo_drawer:
            result_2 = result_1
        else:
            result_2 = self.d2.draw_lines(z)

        return result_1, result_2

    def get_lines(self, img_arr):
        lines_1 = (self.d1.target_names, np.array(self.d1.draw_lines(img_arr)))
        if self.solo_drawer:
            lines_2 = lines_1
        else:
            lines_2 = (self.d2.target_names, np.array(self.d2.draw_lines(img_arr)))

        return lines_1, lines_2

    def get_edges(self, img: np.ndarray):
        # read the image if the `img` is a file name, otherwise treat as a loaded image
        z = img

        result_1 = self.d1.get_edges(z)

        if self.solo_drawer:
            result_2 = result_1
        else:
            result_2 = self.d2.get_edges(z)

        return result_1, result_2


class SubLineDrawer(mp.Process):
    def __init__(
        self, img_queue: mp.Queue, result_queue: mp.Queue, drawer_kwargs: LineDrawerParams
    ):
        super().__init__()
        self.img_queue = img_queue
        self.result_queue = result_queue
        self.drawer = SoloLineDrawer(**drawer_kwargs)

    def run(self):
        while True:
            request = self.img_queue.get()

            if request is None:
                break

            command, img = request

            if command == "draw_lines":
                result = self.drawer.draw_lines(img)
            elif command == "get_edges":
                result = self.drawer.get_edges(img)
            else:
                raise ValueError("Invalid command")

            self.result_queue.put(result)


class ParallelLineDrawer:
    def __init__(
        self,
        angle_kwargs: LineDrawerParams,
        throttle_kwargs: LineDrawerParams,
    ) -> None:
        # self.manager = mp.Manager()

        self.angle_img_q = mp.Queue()
        self.angle_result_q = mp.Queue()
        self.angle_process = SubLineDrawer(
            self.angle_img_q, self.angle_result_q, angle_kwargs
        )

        self.throttle_img_q = mp.Queue()
        self.throttle_result_q = mp.Queue()
        self.throttle_process = SubLineDrawer(
            self.throttle_img_q, self.throttle_result_q, throttle_kwargs
        )

        self.angle_process.start()
        self.throttle_process.start()

        atexit.register(self.cleanup)

    def draw_lines(self, img: Union[str, np.ndarray]):
        # read the image if the `img` is a file name, otherwise treat as a loaded image
        z = cv2.imread(img) if isinstance(img, str) else img

        self.angle_img_q.put(("draw_lines", z))
        self.throttle_img_q.put(("draw_lines", z))

        angle = self.angle_result_q.get()
        throttle = self.throttle_result_q.get()

        return angle, throttle

    def get_lines(self, img_arr):
        angle, throttle = self.draw_lines(img_arr)

        angle_names = self.angle_process.drawer.target_names
        throttle_names = self.throttle_process.drawer.target_names
        return (angle_names, angle), (throttle_names, throttle)

    def get_edges(self, img: np.ndarray):
        self.angle_img_q.put(("get_edges", img))
        self.throttle_img_q.put(("get_edges", img))

        angle = self.angle_result_q.get()
        throttle = self.throttle_result_q.get()

        return angle, throttle

    def cleanup(self):
        self.angle_img_q.put(None)
        self.throttle_img_q.put(None)

        self.angle_process.join()
        self.throttle_process.join()
