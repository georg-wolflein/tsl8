from typing import Callable, TypeVar
from loguru import logger

T = TypeVar("T")


class MPPExtractionError(Exception):
    """Raised when the Microns Per Pixel (MPP) extraction from the slide's metadata fails"""

    pass


class MPPExtractor:
    def __init__(self):
        self.extractors = []

    def register(self, extractor: Callable[[T], float]):
        self.extractors.append(extractor)

    def __call__(self, *args, **kwargs):
        for extractor in self.extractors:
            try:
                slide_mpp = extractor(*args, **kwargs)
                logger.debug(f"MPP successfully extracted using {extractor.__name__}: {slide_mpp:.3f}")
                return slide_mpp
            except MPPExtractionError:
                logger.debug(f"MPP could not be extracted using {extractor.__name__}")
        raise MPPExtractionError("MPP could not be extracted from slide")
