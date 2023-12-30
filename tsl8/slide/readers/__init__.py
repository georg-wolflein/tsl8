from pathlib import Path
from typing import Literal
from loguru import logger

from .readers import SlideReader
from .openslide import OpenSlideReader
from .cucim import CucimReader

Backend = Literal["openslide", "cucim", "auto"]


def make_slide_reader(file: Path, backend: Backend) -> SlideReader:
    if backend == "auto":
        backend = "cucim" if file.suffix in {".svs", ".tiff", ".tif"} else "openslide"
    # logger.debug(f"Using {backend} backend for {file}")
    return {"openslide": OpenSlideReader, "cucim": CucimReader}[backend](file)