import openslide
from openslide import lowlevel as openslide_ll
from ctypes import c_uint32
import numpy as np
from typing import Tuple
from pathlib import Path
from functools import cached_property
import re
from loguru import logger

from .readers import SlideReader
from .mpp import MPPExtractionError, MPPExtractor


class OpenSlideReader(SlideReader):
    def __init__(self, path: Path):
        super().__init__(path)
        self._slide = openslide.OpenSlide(str(path))

    def read_region(self, loc: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        """Adapted from openslide.lowlevel.read_region() to not use PIL images, but directly return a numpy array."""
        x, y = loc
        w, h = size
        if w < 0 or h < 0:
            raise openslide.OpenSlideError("negative width (%d) or negative height (%d) not allowed" % (w, h))
        buf = (w * h * c_uint32)()
        openslide_ll._read_region(self._slide._osr, buf, x, y, level, w, h)
        openslide_ll._convert.argb2rgba(buf)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(*size[::-1], 4)[..., :3]  # remove alpha channel
        return img

    @cached_property
    def mpp(self) -> float:
        return openslide_mpp_extractor(self._slide)

    @property
    def level_dimensions(self) -> Tuple[int, int]:
        return self._slide.level_dimensions

    @property
    def level_downsamples(self) -> Tuple[float]:
        return self._slide.level_downsamples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._slide.close()


openslide_mpp_extractor = MPPExtractor()


@openslide_mpp_extractor.register
def extract_mpp_from_properties(slide: openslide.OpenSlide) -> float:
    try:
        return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except KeyError:
        raise MPPExtractionError


@openslide_mpp_extractor.register
def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    import xml.dom.minidom as minidom

    try:
        xml_path = slide.properties["tiff.ImageDescription"]
    except KeyError:
        raise MPPExtractionError
    try:
        doc = minidom.parseString(xml_path)
    except Exception:
        raise MPPExtractionError
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    if not mpp:
        raise MPPExtractionError
    return mpp


@openslide_mpp_extractor.register
def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    try:
        slide_properties = slide.properties["openslide.comment"]
    except KeyError:
        raise MPPExtractionError
    pattern = r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>"
    match = re.search(pattern, slide_properties)
    if not match:
        raise MPPExtractionError
    return match.group(1)


@openslide_mpp_extractor.register
def extract_mpp_from_xy_and_resolution(slide: openslide.OpenSlide) -> float:
    # https://lists.andrew.cmu.edu/pipermail/openslide-users/2017-April/001385.html
    try:
        unit = slide.properties["tiff.ResolutionUnit"]
        x = float(slide.properties["tiff.XResolution"])
        y = float(slide.properties["tiff.YResolution"])
    except KeyError:
        raise MPPExtractionError

    if x != y:
        logger.warning(f"X and Y resolution are not equal ({x} vs {y}). Using X resolution")

    if unit == "centimeter":
        numerator = 10_000
    elif unit == "inch":
        numerator = 25_400
    else:
        raise MPPExtractionError

    return numerator / x
