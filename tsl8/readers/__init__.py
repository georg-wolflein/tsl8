from pathlib import Path
from typing import Literal, Optional
from loguru import logger

from .readers import SlideReader
from .openslide import OpenSlideReader
from .cucim import CucimReader

Backend = Literal["openslide", "cucim"]


def make_slide_reader(file: Path, backend: Optional[Backend] = None) -> SlideReader:
    file = Path(file)
    if backend is None:
        backend = "cucim" if file.suffix in {".svs", ".tif", ".tiff"} else "openslide"
    logger.debug(f"Using {backend} backend for {file}")
    return {"openslide": OpenSlideReader, "cucim": CucimReader}[backend](file)