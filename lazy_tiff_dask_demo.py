
"""
This demo currently requires the following branch of napari-lazy-openslide
where the cuCIM backend was implemented.

https://github.com/grlee77/napari-lazy-openslide/tree/cucim
"""

import cupy as cp
import dask
import dask.array as da
import numpy as np
import zarr

from napari_lazy_openslide.store import OpenSlideStore

from cucim.skimage import color
from skimage import color as color_cpu

store = OpenSlideStore(
    '/home/lee8rx/Downloads/patient_000/patient_000_node_2.tif',
    tilesize=4096,
)

# data from resolution level "2"
resolution_level = 2
image = da.from_zarr(store, component=resolution_level)
assert image.chunksize[-1] == 3  # required for RGB2HED color conversion

print(f"image.shape = {image.shape}")
print(f"image.chunksize = {image.chunksize}")
print(f"image.numblocks = {image.numblocks}")
on_gpu = True

def rgb2hed_gpu_round_trip(x):
    rgb_gpu = cp.asarray(x, dtype=np.float32)
    hed_gpu = color.rgb2hed(rgb_gpu)
    hed = cp.asnumpy(hed_gpu)
    return hed


def rgb2hed_cpu(x):
    return color_cpu.rgb2hed(x.astype(np.float32))

if on_gpu:
    scheduler = 'single-threaded'
    out = da.map_blocks(
        rgb2hed_gpu_round_trip,
        image,
        dtype="float32",
    )
else:
    scheduler = 'threads'
    out = da.map_blocks(
        rgb2hed_cpu,
        image,
        dtype="float32",
    )

out_file = "/home/lee8rx/hed.zarr"
with dask.config.set(scheduler=scheduler):
    #result = da.to_zarr(out, url=result , compute=True)
    out.to_zarr(out_file, overwrite=True, compute=True)
    # out.compute()
