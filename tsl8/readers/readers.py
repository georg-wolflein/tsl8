from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Sequence, Union
import numpy as np
from pathlib import Path


class SlideReader(ABC):
    def __init__(self, path: Union[Path, str]):
        self._path = Path(path)

    @abstractmethod
    def read_region(self, loc: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        pass

    @abstractproperty
    def mpp(self) -> float:
        pass

    @abstractproperty
    def level_dimensions(self) -> Tuple[int, int]:
        pass

    @abstractproperty
    def level_downsamples(self) -> Sequence[float]:
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Returns the best level for the given downsample."""
        return np.argmin(np.abs(np.array(self.level_downsamples) - downsample))

    def get_best_level_for_mpp(self, target_mpp: float) -> int:
        """Returns the best level for the given mpp."""
        return self.get_best_level_for_downsample(target_mpp / self.mpp)
