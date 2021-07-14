import os


import cupy as cp
import dask
import dask.array as da
import numpy as np
import zarr
from cucim.skimage import restoration as restoration_gpu
from skimage import restoration

"""
100 micron MRI data (downsampled to 200 micron for this demo)

Data download link:
https://openneuro.org/datasets/ds002179/versions/1.1.0

NIFTI-1 format volumes were read in using nibabel and converted to a Zarr array.

References
----------
...[1] Edlow, B.L. et. al. 7 Tesla MRI of the ex vivo human brain at 100 micron
       resolution. Sci Data 6, 244 (2019).
       https://doi.org/10.1038/s41597-019-0254-8
"""

data_dir = '/home/lee8rx/zarr_temp/'
zarr_fname = os.path.join(data_dir, "raw_demo_200um.zarr")

data = da.from_zarr(zarr_fname)

# print shape and chunk properties
print(f"data.shape = {data.shape}")
print(f"data.chunksize = {data.chunksize}")
print(f"data.numblocks = {data.numblocks}")

use_gpu = True

# define a function to denoise a single block on the GPU using cuCIM
def denoise_gpu(x, weight=0.008, eps=2e-4, n_iter_max=50):

    x = cp.asarray(x, dtype=np.float32)
    x = x.mean(0)

    y = restoration_gpu.denoise_tv_chambolle(
        x,
        weight=weight,
        eps=eps,
        n_iter_max=n_iter_max,
    )

    return cp.asnumpy(y)


# define a function to denoise a single block on the CPU using scikit-image
def denoise_cpu(x, weight=0.008, eps=2e-4, n_iter_max=50):

    x = x.astype(np.float32, copy=False).mean(0)

    return restoration.denoise_tv_chambolle(
        x,
        weight=weight,
        eps=eps,
        n_iter_max=n_iter_max,
    )

denoise_func = denoise_gpu if use_gpu else denoise_cpu
scheduler = 'threads'
# scheduler = 'single-threaded'   # use single-threaded to minimize memory use

# apply denoise_gpu over all blocks of the input
denoised = da.map_overlap(
    denoise_func,
    data,
    depth=(0, 1, 1, 1),  # bug fixed in: https://github.com/dask/dask/pull/7894
    boundary='reflect',
    drop_axis=(0,),
    trim=True,
    dtype=np.float32,
)

# write the result of the computation to a new Zarr array
out_file = '/home/lee8rx/zarr_temp/denoised.zarr'
with dask.config.set(scheduler=scheduler):
    denoised.to_zarr(out_file, overwrite=True, compute=True)

# denoised = da.from_zarr(out_file)
