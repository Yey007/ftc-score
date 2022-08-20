from typing import Tuple
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]
Window = Tuple[np.float32, np.float32, np.float32, np.float32]
