import numpy as np
from loguru import logger
import cv2
from tqdm import tqdm
from pathlib import Path
import dask.array as da
import dask
from dask.distributed import as_completed, futures_of
from dask.distributed import Client
from functools import partial
from time import time
from typing import Optional, Union, Literal
import os
import argparse

from tsl8.canny import is_background
from tsl8.slide import load_slide, MPPExtractionError
from tsl8.slide.readers import Backend


SLIDE_EXTENSIONS = [".svs", ".tif", ".dcm", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".tiff", ".svslide", ".bif"]


def reshape_block(block, patch_size):
    h, w, c = block.shape
    block = block.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)

    # Transpose the dimensions to get the tiles in the correct order
    block = block.transpose(0, 2, 1, 3, 4)

    # Reshape again to get a 4D array where each element is a tile
    block = block.reshape(-1, patch_size, patch_size, c)

    return block


def slide_to_chunks(slide, patch_size):
    """Convert a slide to a dask array of chunks.

    Returns: array of dask delayed objects, each of which is a chunk (a 4D array of shape (chunksize, patch_size, patch_size, 3))
    """
    k = np.prod(np.array(slide.chunksize[:2]) // np.array((patch_size, patch_size)))
    d = da.blockwise(
        partial(reshape_block, patch_size=patch_size),
        "kijc",
        slide,
        "ijc",
        dtype=slide.dtype,
        new_axes={"k": k},
        adjust_chunks={"i": patch_size, "j": patch_size},
    )
    patches = d.to_delayed().flatten()
    # patches = da.concatenate(
    #     [da.from_delayed(delayed, shape=(k, patch_size, patch_size, 3), dtype=slide.dtype) for delayed in patches]
    # )

    return patches


def process_block_for_slide(slide_output_dir: Path):
    @dask.delayed
    def process_block(patches, coords):
        """Process a block of patches (perform background rejection, and save to disk)

        Args:
            patches: 4D array of shape (n, patch_size, patch_size, 3)
            coords: 4D array of shape (n, 1, 1, 2)
        """

        # Reject background patches
        patch_mask = is_background(patches)
        patches = patches[~patch_mask]
        coords = coords[~patch_mask].squeeze((1, 2))

        # Save the patches
        for patch, (x, y) in zip(patches, coords):
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(slide_output_dir / f"{x:08d}_{y:08d}.jpg"), patch)

        return patches.shape[0]

    return process_block


def process_slide(
    client: Client,
    slide_file: Path,
    output_path: Path,
    check_status: bool = True,
    patch_size: int = 512,
    target_mpp: float = 0.5,
    level: Optional[int] = 0,  # None means auto select
    patch_to_chunk_multiplier: int = 16,  # 16 means 16x16 per chunk
    backend: Backend = "auto",
    optimize_graph: bool = True,
    description: str = None,
):
    """Process a slide and save the patches to disk."""
    description = description or slide_file.stem

    slide_output_dir = output_path / Path(slide_file).stem
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    start_time = time()
    slide_file = Path(slide_file)

    # Check if the slide has already been processed
    status_file = output_path / "status" / f"{slide_file.stem}.done"
    if check_status and status_file.exists():
        logger.info(f"Skipping {description} because it has already been processed")
        return

    def write_status_file(message: str = "done"):
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with status_file.open("w") as f:
            f.write(message)

    # Load the slide
    try:
        slide = load_slide(
            slide_file,
            target_mpp=target_mpp,
            level=level,
            target_slide_chunk_size=patch_size * patch_to_chunk_multiplier,
            backend=backend,
        )
    except MPPExtractionError:
        logger.warning(f"Could not extract MPP for {slide_file}, skipping")
        write_status_file("mpp_extraction_error")
        raise

    patches = slide_to_chunks(
        slide, patch_size=patch_size
    )  # array of dask delayed objects, each of which is a chunk (a 4D array of shape (n, patch_size, patch_size, 3))

    slide_h, slide_w, _ = slide.shape

    xs, ys = da.meshgrid(da.arange(slide_w // patch_size), da.arange(slide_h // patch_size))
    patch_coords = da.stack([xs, ys], axis=-1) * patch_size
    patch_coords = patch_coords.rechunk((patch_to_chunk_multiplier, patch_to_chunk_multiplier, 2))
    patch_coords = slide_to_chunks(
        patch_coords, patch_size=1
    )  # array of dask delayed objects, each of which is a chunk (a 4D array of shape (n, 1, 1, 2))

    results = [process_block_for_slide(slide_output_dir)(patch, coord) for patch, coord in zip(patches, patch_coords)]
    results = client.persist(results, optimize_graph=optimize_graph)
    for _ in tqdm(
        as_completed(futures_of(results)), total=len(results), desc=f"Processing {description}", unit="chunk"
    ):
        pass

    write_status_file()

    logger.info(f"Finished processing {slide_file.stem} in {time() - start_time:.2f}s")


def process_slides(client: Client, slides_folder: Path, output_path: Path, num_slides: int = None, **kwargs):
    logger.info("Gathering slides")
    slides = sorted(file for file in slides_folder.iterdir() if file.suffix.lower() in SLIDE_EXTENSIONS)
    logger.info(f"Found {len(slides)} slides")

    for i, slide_file in enumerate(slides, 1):
        if num_slides is not None and i > num_slides:
            break
        process_slide(
            client,
            slide_file,
            output_path,
            **kwargs,
            description=f"slide [{i}/{len(slides)}] {slide_file.stem}",
        )

    logger.info(f"Completed processing slides")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="/pathology/camelyon16/training/tumor", type=Path, help="Path to folder containing slides"
    )
    parser.add_argument("--output", default="/data/tsl8_out", type=Path, help="Path to output folder")
    parser.add_argument("--mpp", default=0.5, type=float, help="Target MPP")
    parser.add_argument("--patch-size", default=512, type=int, help="Patch size")
    parser.add_argument("--no-check-status", action="store_false", dest="check_status", help="Don't check status file")
    parser.add_argument(
        "--level",
        default=None,
        type=Optional[int],
        help="Which level of the slide pyramid to use (if unspecified, select the highest level with MPP lower than the target MPP)",
    )
    parser.add_argument(
        "--patch-to-chunk-multiplier",
        default=16,
        type=int,
        help="How many patches to put in each chunk (this value will be squared to get the total number of patches per chunk)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        type=str,
        choices=["auto", "openslide", "cucim"],
        help="Which backend to use for reading the slide (set to 'auto' to select the best backend based on the file extension)",
    )
    parser.add_argument(
        "--no-optimize-graph",
        action="store_false",
        dest="optimize_graph",
        help="Don't optimize the dask graph before processing",
    )
    parser.add_argument("--workers", default=32, type=int, help="Number of processes")
    parser.add_argument("--threads-per-worker", default=1, type=int, help="Number of threads per process")
    parser.add_argument("--memory-limit", default="64GB", type=str, help="Memory limit per process")
    parser.add_argument(
        "--n", default=None, type=int, help="Number of slides to process (for debugging); default is all"
    )

    parser.set_defaults(check_status=True, optimize_graph=True)
    args = parser.parse_args()

    client = Client(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
    )

    logger.info(f"Started Dask client with {len(client.scheduler_info()['workers'])} workers")
    logger.info(f"Access the Dask dashboard at {client.dashboard_link}")

    process_slides(
        client=client,
        slides_folder=args.input,
        output_path=args.output,
        check_status=args.check_status,
        patch_size=args.patch_size,
        target_mpp=args.mpp,
        level=args.level,
        patch_to_chunk_multiplier=args.patch_to_chunk_multiplier,
        backend=args.backend,
        optimize_graph=args.optimize_graph,
        num_slides=args.n,
    )


if __name__ == "__main__":
    main()
