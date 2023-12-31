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
import sys

from tsl8.canny import is_background
from tsl8.slide import Backend, make_slide_reader, MPPExtractionError


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
    n_threads_per_slide: int = 16,
    tqdm_position: int = 0,
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
                int(np.ceil(x / loaded_slide_chunk_size)) * loaded_slide_chunk_size,
                int(np.ceil(y / loaded_slide_chunk_size)) * loaded_slide_chunk_size,
            )

            x_chunks, y_chunks = level_dim_x // loaded_slide_chunk_size, level_dim_y // loaded_slide_chunk_size
            logger.debug(
                f"Level {level} is of size {x}x{y} (padded to {level_dim_x}x{level_dim_y}) and will be split into {x_chunks}x{y_chunks} chunks each of size {loaded_slide_chunk_size}x{loaded_slide_chunk_size}, resized to {target_slide_chunk_size}x{target_slide_chunk_size}"
            )

            def load_chunk(chunk_x_index, chunk_y_index):
                # Compute chunk location in target MPP coordinates
                x, y = chunk_x_index * target_slide_chunk_size, chunk_y_index * target_slide_chunk_size

                # Compute chunk location in slide coordinates (slide_mpp is level0_mpp)
                level0_x = int(x * target_mpp / slide_mpp)
                level0_y = int(y * target_mpp / slide_mpp)

                size = (loaded_slide_chunk_size, loaded_slide_chunk_size)
                # print("read", chunk_x, chunk_y, location, level, size)
                chunk = slide.read_region((level0_x, level0_y), level, size)
                chunk = cv2.resize(chunk, (target_slide_chunk_size, target_slide_chunk_size))
                return chunk, (x, y)

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

            with ThreadPoolExecutor(max_workers=n_threads_per_slide) as executor:
                futures = [
                    executor.submit(process_chunk, chunk_x_index, chunk_y_index)
                    for chunk_x_index in range(x_chunks)
                    for chunk_y_index in range(y_chunks)
                ]
                for _ in tqdm(
                    as_completed(futures),
                    desc=f"Worker #{tqdm_position:02d}: {description}",
                    unit="chunk",
                    total=x_chunks * y_chunks,
                    position=tqdm_position,
                    leave=False,
                ):
                    pass
    except MPPExtractionError:
        logger.warning(f"Could not extract MPP for {slide_file}, skipping")
        write_status_file("mpp_extraction_error")
        raise

    write_status_file()
    logger.debug(f"Finished processing {slide_file.stem} in {time() - start_time:.2f}s")


def process_slides(slides_folder: Path, output_path: Path, num_slides: int = None, n_workers: int = 4, **kwargs):
    logger.info("Gathering slides")
    slides = sorted(file for file in slides_folder.iterdir() if file.suffix.lower() in SLIDE_EXTENSIONS)
    logger.info(f"Found {len(slides)} slides")

    def worker(input_queue, output_queue, worker_id: int):
        while True:
            item = input_queue.get()
            if item is None:
                break
            i, slide_file = item
            process_slide(
                slide_file,
                output_path,
                **kwargs,
                description=f"slide [{i}/{num_slides}] {slide_file.stem}",
                tqdm_position=worker_id,
            )
            output_queue.put(i)

    # Enqueue slides
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    tasks = []
    for i, slide_file in enumerate(slides, 1):
        if num_slides is not None and i > num_slides:
            break
        args = i, slide_file
        tasks.append(args)
        input_queue.put(args)
    num_slides = len(tasks)

    # Start workers
    start = time()
    workers = [mp.Process(target=worker, args=(input_queue, output_queue, i + 1)) for i in range(n_workers)]
    for w in workers:
        w.start()

    # Wait for workers to finish
    for _ in tqdm(range(num_slides), desc="Processing slides", unit="slide", position=0, leave=False):
        output_queue.get()

    # Stop workers
    for _ in workers:
        input_queue.put(None)

    logger.info(
        f"Completed processing {num_slides} slides in {time() - start:.2f}s ({(time() - start) / num_slides:.2f}s/slide)"
    )


def main():
    parser = argparse.ArgumentParser("tsl8", description="Extract patches from slides")
    parser.add_argument("--input", default=None, type=Path, help="Path to folder containing slides")
    parser.add_argument("--output", default=None, type=Path, help="Path to output folder")
    parser.add_argument("--mpp", default=0.5, type=float, help="Target MPP")
    parser.add_argument("--patch-size", "-p", default=512, type=int, help="Patch size")
    parser.add_argument(
        "--no-skip", action="store_false", dest="check_status", help="Don't skip slides based on status file"
    )
    parser.add_argument(
        "--level",
        default=None,
        type=Optional[int],
        help="Which level of the slide pyramid to use (if unspecified, select the highest level with MPP lower than the target MPP)",
    )
    parser.add_argument(
        "--patch-to-chunk-multiplier",
        "-k",
        default=24,
        type=int,
        help="How many patches to put in each chunk (this value will be squared to get the total number of patches per chunk)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        default="auto",
        type=str,
        choices=["auto", "openslide", "cucim"],
        help="Which backend to use for reading the slide (set to 'auto' to select the best backend based on the file extension)",
    )
    parser.add_argument(
        "--workers", "-w", default=4, type=int, help="Number of slides to process in parallel (one process per slide)"
    )
    parser.add_argument(
        "--threads-per-worker",
        "-t",
        default=16,
        type=int,
        help="Number of threads per slide/process (how many chunks to process in parallel per slide)",
    )
    parser.add_argument(
        "--num-slides", "-n", default=None, type=int, help="Number of slides to process (for debugging); default is all"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.set_defaults(check_status=True, optimize_graph=True, debug=False)
    args = parser.parse_args()

    if args.input is None or args.output is None:
        parser.error("--input and --output options are required")

    # Set up logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if args.debug else "INFO",
    )

    # Process slides
    process_slides(
        slides_folder=args.input,
        output_path=args.output,
        check_status=args.check_status,
        patch_size=args.patch_size,
        target_mpp=args.mpp,
        level=args.level,
        patch_to_chunk_multiplier=args.patch_to_chunk_multiplier,
        backend=args.backend,
        num_slides=args.num_slides,
        n_workers=args.workers,
        n_threads_per_slide=args.threads_per_worker,
    )


if __name__ == "__main__":
    main()
