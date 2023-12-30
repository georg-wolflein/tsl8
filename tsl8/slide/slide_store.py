"""
This code adapted from https://github.com/manzt/napari-lazy-openslide/blob/main/napari_lazy_openslide/store.py.

Main changes:
  - support serialization, as per https://github.com/crs4/napari-lazy-openslide/blob/main/napari_lazy_openslide/store.py, a pull request that never got merged
  - more efficient loading using lower-level openslide APIs
  - use RGB channels instead of RGBA
"""

from ctypes import ArgumentError
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple
import numpy as np
from openslide import OpenSlide
from zarr.storage import _path_to_prefix, attrs_key, init_array, init_group, BaseStore
from zarr.util import json_dumps, normalize_storage_path

from .readers import SlideReader, make_slide_reader


def init_attrs(store: MutableMapping, attrs: Mapping[str, Any], path: str = None):
    path = normalize_storage_path(path)
    path = _path_to_prefix(path)
    store[path + attrs_key] = json_dumps(attrs)


def create_meta_store(
    slide: SlideReader, tilesize: int, level_dimensions: Sequence[Tuple[int, int]]
) -> Dict[str, bytes]:
    """Creates a dict containing the zarr metadata for the multiscale openslide image."""
    store = dict()
    root_attrs = {
        "multiscales": [
            {
                "name": slide._path.name,
                "datasets": [{"path": str(i)} for i in range(len(level_dimensions))],
                "version": "0.1",
            }
        ]
    }
    init_group(store)
    init_attrs(store, root_attrs)
    for i, (x, y) in enumerate(level_dimensions):
        init_array(
            store,
            path=str(i),
            shape=(y, x, 3),
            chunks=(tilesize, tilesize, 3),
            dtype="|u1",
            compressor=None,
        )
    return store


def _parse_chunk_path(path: str):
    """Returns x,y chunk coords and pyramid level from string key"""
    level, ckey = path.split("/")
    y, x, _ = map(int, ckey.split("."))
    return x, y, int(level)


def _pad_level_dimensions(level_dimensions: Sequence[Tuple[int, int]], tilesize: int) -> Sequence[Tuple[int, int]]:
    """Pad level dimensions to be multiples of the tilesize."""
    return [(int(np.ceil(x / tilesize)) * tilesize, int(np.ceil(y / tilesize)) * tilesize) for x, y in level_dimensions]


class SlideStore(BaseStore):
    """Wraps a SlideReader object as a multiscale Zarr Store.

    Parameters
    ----------
    path: str
        The file to open with OpenSlide.
    tilesize: int
        Desired "chunk" size for zarr store.
    """

    _readable = True
    _writeable = False
    _erasable = False
    _listable = True
    _store_version = 2

    def __init__(self, path: str, tilesize: int = 512, pad: bool = True, backend: str = "auto"):
        self._path = path
        self._slide = make_slide_reader(path, backend=backend)
        self._tilesize = tilesize
        self._level_dimensions = (
            _pad_level_dimensions(self._slide.level_dimensions, tilesize) if pad else self._slide.level_dimensions
        )
        self._store = create_meta_store(self._slide, tilesize, self._level_dimensions)

    def __getitem__(self, key: str):
        if key in self._store:
            # key is for metadata
            return self._store[key]

        # key should now be a path to an array chunk
        # e.g '3/4.5.0' -> '<level>/<chunk_key>'
        try:
            x, y, level = _parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = (self._tilesize, self._tilesize)
            # print("read", x, y, location, level, size)
            # tile = np.array(self._slide.read_region(location, level, size))[..., :3]
            tile = self._slide.read_region(location, level, size)
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            raise err
        except Exception:
            # TODO: probably need better error handling.
            # If anything goes wrong, we just signal the chunk
            # is missing from the store.
            raise KeyError(key)

        return tile.tobytes()

    # NOTE: do NOT implement __contains__ (this caused me a ~3h headache)!

    def __contains__(self, key):
        if key in self._store:
            return True
        try:
            x, y, level = _parse_chunk_path(key)
            if x < 0 or y < 0:
                return False
            if not (0 <= level <= len(self._level_dimensions)):
                return False
            lvl_x, lvl_y = self._level_dimensions[level]
            return x <= lvl_x and y <= lvl_y
        except Exception:
            return False

    def __eq__(self, other):
        return isinstance(other, SlideStore) and self._slide._filename == other._slide._filename

    def __setitem__(self, key, val):
        raise RuntimeError("__setitem__ not implemented")

    def __delitem__(self, key):
        raise RuntimeError("__setitem__ not implemented")

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(1 for _ in self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _ref_pos(self, x: int, y: int, level: int):
        dsample = self._slide.level_downsamples[level]
        xref = int(x * dsample * self._tilesize)
        yref = int(y * dsample * self._tilesize)
        return xref, yref

    # def keys(self):
    #     return self._store.keys()

    def close(self):
        self._slide.close()

    def __getstate__(self):
        return {"_path": self._path, "_tilesize": self._tilesize}

    def __setstate__(self, newstate):
        path = newstate["_path"]
        tilesize = newstate["_tilesize"]
        self.__init__(path, tilesize)


if __name__ == "__main__":
    import sys

    store = SlideStore(sys.argv[1])
