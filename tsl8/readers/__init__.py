from pathlib import Path
from typing import Literal

from .readers import SlideReader
from .openslide import OpenSlideReader
from .cucim import CucimReader

Backend = Literal["openslide", "cucim"]


def make_slide_reader(file: Path, backend: Backend) -> SlideReader:
    return {"openslide": OpenSlideReader, "cucim": CucimReader}[backend](file)
