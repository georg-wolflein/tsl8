tsl8
====

*tsl8*, pronounced "tesselate" is a parallelised solution for reading whole slide images (WSIs), splitting them into non-overlapping patches/tiles, and rejecting background tiles.

# Installation
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Running
```
source env/bin/activate
python -m tsl8 --input ... --output ...
```
Note: use `--help` for a list of options. 

# Features
- **Slide backends.** Supports `cucim` and `openslide` (controlled via the `--backend` option). By default, it will choose the fasted available backend based on the slide's file type.
- **Level selection.** Intelligently selects the highest level from the image pyramid with a level MPP lower than the target MPP (use `--level` to force a specific level).
- **Multiprocessing.** Process multiple slides in parallel, each slide in a separate process (use `--workers` option to control number of slides to process in parallel)
- **Multithreading.** Each process spawns muliple threads (controlled via the `--threads-per-worker` option) to tesselate the slide. 
- **Chunking.** Reads and writes are performed in chunks. Each chunk contains $k^2$ patches ($k$ is set via the `--patch-to-chunk-multiplier` option).
- **Optimised memory and computation.** Because we operate on chunks, we do not need to load the whole slide into memory. At any given time, for $w$ workers and $t$ threads, a maximum of $wt$ chunks are loaded in memory, where each chunk contains $k^2$ patches (patch size is controlled using the `--patch-size` argument).
- **Background rejection.** Background tiles are rejected (not saved).
- **Intelligent resuming.** Slides that completed processing will not be processed again when re-running the program.

# Usage
```
$ python -m tsl8 --help
usage: tsl8 [-h] [--input INPUT] [--output OUTPUT] [--mpp MPP] [--patch-size PATCH_SIZE] [--no-check-status] [--level LEVEL]
            [--patch-to-chunk-multiplier PATCH_TO_CHUNK_MULTIPLIER] [--backend {auto,openslide,cucim}] [--workers WORKERS]
            [--threads-per-worker THREADS_PER_WORKER] [--num-slides NUM_SLIDES] [--debug]

Extract patches from slides

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to folder containing slides
  --output OUTPUT       Path to output folder
  --mpp MPP             Target MPP
  --patch-size PATCH_SIZE, -p PATCH_SIZE
                        Patch size
  --no-check-status     Don't check status file
  --level LEVEL         Which level of the slide pyramid to use (if unspecified, select the highest level with MPP lower than the target
                        MPP)
  --patch-to-chunk-multiplier PATCH_TO_CHUNK_MULTIPLIER, -k PATCH_TO_CHUNK_MULTIPLIER
                        How many patches to put in each chunk (this value will be squared to get the total number of patches per chunk)
  --backend {auto,openslide,cucim}, -b {auto,openslide,cucim}
                        Which backend to use for reading the slide (set to 'auto' to select the best backend based on the file extension)
  --workers WORKERS, -w WORKERS
                        Number of slides to process in parallel (one process per slide)
  --threads-per-worker THREADS_PER_WORKER, -t THREADS_PER_WORKER
                        Number of threads per slide/process (how many chunks to process in parallel per slide)
  --num-slides NUM_SLIDES, -n NUM_SLIDES
                        Number of slides to process (for debugging); default is all
  --debug               Enable debug logging
```