import dask.array as da
import zarr
from pathlib import Path
from typing import Union
import numpy as np
import cv2
from typing import Optional
from loguru import logger

from .slide_store import SlideStore
from .readers import make_slide_reader, Backend
from .readers.mpp import MPPExtractionError


def load_slide(
    slide_file: Union[Path, str],
    target_mpp: float = 256.0 / 224.0,
    level: Optional[int] = None,
    backend: Backend = "auto",
    target_slide_chunk_size: int = 224 * 16,
) -> da.Array:
    with make_slide_reader(slide_file, backend=backend) as slide:
        slide_mpp = slide.mpp

        logger.debug(
            f"Slide has {slide.level_count} levels with following downsamples: {({k: v for k, v in enumerate(slide.level_downsamples)})}"
        )

        # Intelligently choose correct level
        proposed_level = slide.get_best_level_for_downsample(target_mpp / slide_mpp)
        if level is None:
            level = proposed_level
        elif proposed_level < level:
            logger.warning(
                f"The requested slide input level {level} is too high for {target_mpp=:.3f} and {slide_mpp=:.3f}; using level {proposed_level} instead."
            )
            level = proposed_level

        level_mpp = slide.level_downsamples[level] * slide_mpp
        logger.debug(f"Using level {level} with {level_mpp=:.3f} for {slide_mpp=:.3f} and {target_mpp=:.3f}")

    loaded_slide_chunk_size = np.ceil(target_slide_chunk_size * target_mpp / level_mpp).astype(int)

    def resize_chunk(chunk):
        result = cv2.resize(chunk, (target_slide_chunk_size, target_slide_chunk_size))
        return result

    store = SlideStore(slide_file, tilesize=loaded_slide_chunk_size, pad=True, backend=backend)
    grp = zarr.open(store, mode="r")

    z = grp[level]

    dz = da.from_zarr(z)
    dz_resized = da.map_blocks(
        resize_chunk, dz, chunks=(target_slide_chunk_size, target_slide_chunk_size, 3), meta=np.empty(1), dtype=np.uint8
    )

    return dz_resized
