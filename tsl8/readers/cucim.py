from cucim import CuImage
from ctypes import c_uint32
import numpy as np
from typing import Tuple
from pathlib import Path
from skimage.util import img_as_float
from functools import cached_property

from .readers import SlideReader
from .mpp import MPPExtractionError, MPPExtractor


class CucimReader(SlideReader):
    def __init__(self, path: Path):
        super().__init__(path)
        self._slide = CuImage(str(path))

    def read_region(self, loc: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        region = self._slide.read_region(loc, size, level)
        return ((img_as_float(np.asarray(region))) * 255).astype(np.uint8)

    @cached_property
    def mpp(self) -> float:
        # TODO: implement mpp extraction for cucim
        from .openslide import openslide_mpp_extractor
        from openslide import OpenSlide

        with OpenSlide(str(self._path)) as slide:
            return openslide_mpp_extractor(slide)

    @property
    def level_dimensions(self) -> Tuple[int, int]:
        return self._slide.resolutions["level_dimensions"]

    @property
    def level_downsamples(self) -> Tuple[float]:
        return self._slide.resolutions["level_downsamples"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._slide.close()
