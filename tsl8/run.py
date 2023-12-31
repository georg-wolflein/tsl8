import numpy as np
from loguru import logger
import cv2
from tqdm import tqdm
from pathlib import Path
from functools import partial
from time import time
from typing import Optional, Union, Literal
import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tsl8.canny import is_background
from tsl8.slide import load_slide, MPPExtractionError
from tsl8.slide.readers import Backend, make_slide_reader


SLIDE_EXTENSIONS = [".svs", ".tif", ".dcm", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".tiff", ".svslide", ".bif"]


def process_slide(
    slide_file: Path,
    output_path: Path,
    check_status: bool = True,
    patch_size: int = 512,
    target_mpp: float = 0.5,
    level: Optional[int] = 0,  # None means auto select
    patch_to_chunk_multiplier: int = 16,  # 16 means 16x16 per chunk
    backend: Backend = "auto",
    description: str = None,
    n_threads_per_slide: int = 32,
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
    target_slide_chunk_size = patch_size * patch_to_chunk_multiplier
    try:
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

            # Compute padded level dimensions for selected level
            x, y = slide.level_dimensions[level]
            level_dim_x, level_dim_y = (
                int(np.ceil(x / target_slide_chunk_size)) * target_slide_chunk_size,
                int(np.ceil(y / target_slide_chunk_size)) * target_slide_chunk_size,
            )

            x_chunks, y_chunks = level_dim_x // target_slide_chunk_size, level_dim_y // target_slide_chunk_size
            logger.debug(f"Level {level} will be split into {x_chunks}x{y_chunks} chunks")

            def _ref_pos(x: int, y: int, level: int):
                dsample = slide.level_downsamples[level]
                xref = int(x * dsample * patch_size)
                yref = int(y * dsample * patch_size)
                return xref, yref

            def load_chunk(chunk_x_index, chunk_y_index):
                chunk_x, chunk_y = chunk_x_index * target_slide_chunk_size, chunk_y_index * target_slide_chunk_size
                location = _ref_pos(chunk_x, chunk_y, level)
                size = (loaded_slide_chunk_size, loaded_slide_chunk_size)
                # print("read", chunk_x, chunk_y, location, level, size)
                chunk = slide.read_region(location, level, size)
                chunk = cv2.resize(chunk, (target_slide_chunk_size, target_slide_chunk_size))
                return chunk, (chunk_x, chunk_y)

            def process_chunk(chunk_x_index, chunk_y_index):
                # Load chunk
                chunk, (chunk_x, chunk_y) = load_chunk(chunk_x_index, chunk_y_index)
                width, height = chunk.shape[:2]

                for x_offset in range(0, width, patch_size):
                    for y_offset in range(0, height, patch_size):
                        patch = chunk[y_offset : y_offset + patch_size, x_offset : x_offset + patch_size]
                        # Reject background patches
                        if is_background(patch):
                            continue
                        # Save the patch
                        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(
                            str(slide_output_dir / f"{chunk_x + x_offset:08d}_{chunk_y + y_offset:08d}.jpg"), patch
                        )

            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [
                    executor.submit(process_chunk, chunk_x_index, chunk_y_index)
                    for chunk_x_index in range(x_chunks)
                    for chunk_y_index in range(y_chunks)
                ]
                for _ in tqdm(
                    as_completed(futures),
                    desc=f"Processing {description}",
                    unit="chunk",
                    total=x_chunks * y_chunks,
                ):
                    pass
    except MPPExtractionError:
        logger.warning(f"Could not extract MPP for {slide_file}, skipping")
        write_status_file("mpp_extraction_error")
        raise

    write_status_file()
    logger.info(f"Finished processing {slide_file.stem} in {time() - start_time:.2f}s")


def process_slides(slides_folder: Path, output_path: Path, num_slides: int = None, **kwargs):
    logger.info("Gathering slides")
    slides = sorted(file for file in slides_folder.iterdir() if file.suffix.lower() in SLIDE_EXTENSIONS)
    logger.info(f"Found {len(slides)} slides")

    for i, slide_file in enumerate(slides, 1):
        if num_slides is not None and i > num_slides:
            break
        process_slide(
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
        default=24,
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
    parser.add_argument("--workers", default=32, type=int, help="Number of processes")
    parser.add_argument("--threads-per-worker", default=1, type=int, help="Number of threads per process")
    parser.add_argument("--memory-limit", default="64GB", type=str, help="Memory limit per process")
    parser.add_argument(
        "--n", default=None, type=int, help="Number of slides to process (for debugging); default is all"
    )

    parser.set_defaults(check_status=True, optimize_graph=True)
    args = parser.parse_args()

    process_slides(
        slides_folder=args.input,
        output_path=args.output,
        check_status=args.check_status,
        patch_size=args.patch_size,
        target_mpp=args.mpp,
        level=args.level,
        patch_to_chunk_multiplier=args.patch_to_chunk_multiplier,
        backend=args.backend,
        num_slides=args.n,
    )


if __name__ == "__main__":
    main()
