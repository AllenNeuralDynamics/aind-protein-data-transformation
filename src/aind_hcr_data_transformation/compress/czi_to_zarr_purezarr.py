"""Pure zarr v3 + OME-NGFF 0.5 writer backend (simplified).

This version follows the SmartSPIM pattern for simplicity and maintainability.
Always uses S3, zarr v3, Blosc compression, dask, and xarray_multiscale.
"""

import logging
import os
import time
from typing import Any, List, Optional

import czifile
import dask.array as da
from dask.array.core import from_array
import numpy as np
import s3fs
import xarray_multiscale
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url

from aind_hcr_data_transformation.compress.omezarr_metadata import (
    _get_pyramid_metadata,
    write_ome_ngff_metadata,
)
from aind_hcr_data_transformation.utils.utils import (
    czi_block_generator,
    pad_array_n_d,
)


class BlockedArrayWriter:
    """Simple blocked array writer for large chunks."""
    
    @staticmethod
    def get_block_shape(arr: Any, target_size_mb: int = 12800) -> tuple:
        """Get optimal block shape for writing, similar to SmartSPIM pattern."""
        target_bytes = target_size_mb * 1024**2
        itemsize = arr.dtype.itemsize
        
        # Start with chunk shape and expand
        chunk_shape = list(arr.chunksize)
        spatial_dims = chunk_shape[-3:]  # Z, Y, X
        
        # Calculate current size
        current_size = np.prod(spatial_dims) * itemsize
        
        if current_size >= target_bytes:
            return tuple(spatial_dims)
            
        # Scale up proportionally
        scale_factor = (target_bytes / current_size) ** (1/3)
        scaled = [min(int(dim * scale_factor), shape_dim) 
                 for dim, shape_dim in zip(spatial_dims, arr.shape[-3:])]
        
        return tuple(scaled)
    
    @staticmethod
    def store(dask_array: Any, zarr_array: Any, block_shape: tuple):
        """Store dask array into zarr array using blocked writes."""
        if len(block_shape) != 3:
            raise ValueError("block_shape must be (z, y, x)")
        
        z_block, y_block, x_block = block_shape
        shape = dask_array.shape
        
        # Iterate over spatial blocks only (T,C dimensions written fully each time)
        for z in range(0, shape[-3], z_block):
            z_end = min(z + z_block, shape[-3])
            for y in range(0, shape[-2], y_block):
                y_end = min(y + y_block, shape[-2])
                for x in range(0, shape[-1], x_block):
                    x_end = min(x + x_block, shape[-1])
                    
                    # Full T,C slices with spatial block
                    region = (slice(None), slice(None), 
                             slice(z, z_end), slice(y, y_end), slice(x, x_end))
                    
                    block = dask_array[region].compute()
                    zarr_array[region] = block


def safe_create_zarr_group(store, path: str = "") -> zarr.Group:
    """Safely create or open a zarr group."""
    try:
        return zarr.open_group(store, path=path, mode="r+")
    except (ValueError, KeyError):
        return zarr.group(store, path=path, overwrite=False)


def compute_pyramid(
    data: Any,
    n_lvls: int,
    scale_factors: tuple,
    chunks: Any = "auto"
) -> List[Any]:
    """Compute pyramid levels using xarray_multiscale."""
    # Create a wrapper to handle different reducer signatures
    def windowed_mean_wrapper(array, window_size, **kwargs):
        return xarray_multiscale.reducers.windowed_mean(array, tuple(window_size), **kwargs)
    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=windowed_mean_wrapper,
        scale_factors=scale_factors,
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]
    
    return [level.data for level in pyramid]


def _split_s3_path(path: str) -> tuple[str, str]:
    """Split s3://bucket/key into (bucket, key)."""
    if not path.startswith("s3://"):
        raise ValueError("Path must start with s3://")
    
    no_scheme = path[5:]
    bucket, _, key = no_scheme.partition("/")
    return bucket, key.rstrip("/")


def _dataset_exists(bucket: str, key: str) -> bool:
    """Check if dataset already exists by looking for zarr.json."""
    fs = s3fs.S3FileSystem(anon=False)
    return fs.exists(f"{bucket}/{key}/zarr.json")

def _ensure_scale_factor_5D(scale_factor: List[int]) -> List[int]:
    """Ensure scale_factor is a 5D list (T,C,Z,Y,X)."""
    if len(scale_factor) > 5:
        raise ValueError("scale_factor cannot have more than 5 dimensions")
    return ([1] * (5 - len(scale_factor))) + scale_factor

def czi_stack_zarr_writer(
    czi_path: str,
    output_path: str,
    voxel_size: List[float],
    shard_size: List[int],  # Kept for API compatibility (ignored)
    chunk_size: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    stack_name: str,
    compressor_kwargs: dict,
    bucket_name: Optional[str] = None,
    downsample_mode: Optional[str] = "mean",  # Only mean supported
    overwrite_existing_data: bool = False,
    macro_block_target_mb: int = 12800,
):
    """
    Write CZI stack to OME-Zarr format using pure zarr v3 (simplified implementation).
    
    This version is simplified for maintainability and follows SmartSPIM patterns:
    - Always uses S3 storage
    - Always uses Blosc compression  
    - Always builds pyramid with xarray_multiscale
    - Simple blocked writing strategy
    """
    
    # Handle S3 path normalization
    if output_path.startswith("s3://"):
        bucket, key_base = _split_s3_path(output_path)
        if bucket_name and bucket_name != bucket:
            logging.warning(f"bucket_name {bucket_name} differs from path bucket {bucket}. Using {bucket}")
        bucket_name = bucket
        output_path = key_base
    
    if bucket_name is None:
        raise ValueError("bucket_name must be provided or embedded in s3:// path")
    
    # Construct dataset path
    dataset_path = f"{output_path.strip('/')}/{stack_name}"
    
    # Check for existing dataset
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_JOB_ID")
    if _dataset_exists(bucket_name, dataset_path) and not overwrite_existing_data:
        logging.info(f"[purezarr]{' [SLURM '+slurm_id+']' if slurm_id else ''} "
                    f"Skipping existing dataset at s3://{bucket_name}/{dataset_path}")
        return
    
    start_time = time.time()
    
    # Log ignored parameters for clarity
    if shard_size:
        logging.debug("shard_size parameter ignored (using chunk_size)")
    if downsample_mode and downsample_mode != "mean":
        logging.warning(f"Only mean downsampling supported, ignoring {downsample_mode}")
    
    with czifile.CziFile(str(czi_path)) as czi:
        # Get 5D shape (T,C,Z,Y,X) - pad with 1s as needed  
        # Access shape through asarray to avoid lazyattr issues
        sample_data = czi.asarray()[:1]  # Get just first element to get shape
        czi_shape = sample_data.shape[:-len(sample_data.shape)+len(czi.asarray().shape)]
        czi_shape = czi.asarray().shape  # Get full shape
        shape_tuple = tuple(int(i) for i in czi_shape if i != 1)
        extra_dims = (1,) * (5 - len(shape_tuple))
        dataset_shape = extra_dims + shape_tuple
        
        # Ensure chunk_size is 5D
        chunk_size_5d = ([1] * (5 - len(chunk_size))) + chunk_size
        
        base_dtype = np.dtype(str(czi.dtype))
        
        logging.info(f"[purezarr] Writing {stack_name} to s3://{bucket_name}/{dataset_path}")
        logging.info(f"[purezarr] Dataset shape: {dataset_shape}, chunks: {chunk_size_5d}")
        
        # Channel metadata for OMERO
        np_info_func = np.iinfo if np.issubdtype(base_dtype, np.integer) else np.finfo
        type_info = np_info_func(base_dtype)
        channel_minmax = [(float(type_info.min), float(type_info.max)) 
                         for _ in range(dataset_shape[1])]
        channel_startend = [(90.0, 1200.0) for _ in range(dataset_shape[1])]
        
        # Create S3 store and root group
        s3_url = f"s3://{bucket_name}/{dataset_path}"
        parsed_result = parse_url(path=s3_url, mode="w")
        if parsed_result is None:
            raise ValueError(f"Failed to parse S3 URL: {s3_url}")
        store = parsed_result.store
        root_group = safe_create_zarr_group(store=store)
        
        # Create compressor
        compressor = Blosc(**compressor_kwargs)
        
        # Create subgroup for this stack
        stack_group = root_group.create_group(name=stack_name.split('/')[-1], overwrite=True)
        
        # Create level 0 array
        arr0 = stack_group.create_dataset(
            name="0",
            shape=dataset_shape,
            chunks=tuple(chunk_size_5d),
            dtype=base_dtype,
            compressor=compressor,
            overwrite=True,
        )
        
        # Write level 0 data using streaming approach
        logging.info("[purezarr] Writing level 0 data...")
        z_jump = chunk_size_5d[-3]
        total_regions = 0
        
        for z_block, axis_area in czi_block_generator(czi, axis_jumps=z_jump, slice_axis="z"):
            z_block = pad_array_n_d(z_block)
            z_block = np.asarray(z_block)
            
            z_start, z_stop = axis_area.start, axis_area.stop
            local_z = z_stop - z_start
            
            # Write in chunk-sized pieces for Y,X
            stride_y, stride_x = chunk_size_5d[-2], chunk_size_5d[-1]
            
            for y0 in range(0, dataset_shape[-2], stride_y):
                y1 = min(y0 + stride_y, dataset_shape[-2])
                for x0 in range(0, dataset_shape[-1], stride_x):
                    x1 = min(x0 + stride_x, dataset_shape[-1])
                    
                    region = (slice(0, dataset_shape[0]), slice(0, dataset_shape[1]),
                             slice(z_start, z_stop), slice(y0, y1), slice(x0, x1))
                    
                    # z_block is 5D after pad_array_n_d: (T, C, Z, Y, X)
                    # Extract matching sub_block for this spatial region
                    sub_block = z_block[:, :, 0:local_z, y0:y1, x0:x1]
                    arr0[region] = sub_block
                    total_regions += 1
                    
                    if total_regions % 100 == 0:
                        logging.debug(f"[purezarr] Wrote {total_regions} regions")
        
        logging.info(f"[purezarr] Finished level 0: {total_regions} regions written")
        
        # Build pyramid if needed
        written_arrays = [arr0]
        if n_lvls > 1 and dataset_shape[2] > 1:
            logging.info("[purezarr] Building pyramid...")
            
            # Create dask array from level 0
            arr0_data = np.array(arr0[:])  # Load the data into memory first
            arr0_dask = from_array(arr0_data, chunks=chunk_size_5d)  # type: ignore
            
            # Ensure scale_factor is 5D
            scale_factor = _ensure_scale_factor_5D(scale_factor)
            # Compute pyramid
            pyramid_data = compute_pyramid(
                data=arr0_dask,
                n_lvls=n_lvls,
                scale_factors=tuple(scale_factor), 
                chunks=arr0_dask.chunksize
            )
            
            # Write pyramid levels (skip level 0 since it's already written)
            for level_idx in range(1, len(pyramid_data)):
                level_data = pyramid_data[level_idx]
                level_shape = level_data.shape
                level_chunks = tuple(min(c, s) for c, s in zip(chunk_size_5d, level_shape))
                
                # Create zarr array for this level
                arr_level = stack_group.create_dataset(
                    name=str(level_idx),
                    shape=level_shape,
                    chunks=level_chunks,
                    dtype=base_dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                
                # Write using blocked writer
                block_shape = BlockedArrayWriter.get_block_shape(
                    level_data, target_size_mb=macro_block_target_mb
                )
                
                logging.info(f"[purezarr] Writing level {level_idx}, block shape: {block_shape}")
                BlockedArrayWriter.store(level_data, arr_level, block_shape)
                written_arrays.append(arr_level)
        
        # Write OME-NGFF metadata to the stack group
        metadata_dict = write_ome_ngff_metadata(
            arr_shape=list(dataset_shape),
            chunk_size=chunk_size_5d,
            image_name=stack_name,
            n_lvls=n_lvls,
            scale_factors=tuple(scale_factor),
            voxel_size=tuple(voxel_size),
            channel_names=[channel_name],
            channel_colors=None,
            channel_minmax=channel_minmax,
            channel_startend=channel_startend,
            metadata=_get_pyramid_metadata(),
            origin=[0, 0, 0],
        )
        
        # Apply metadata to stack group
        if "attributes" in metadata_dict:
            for key, value in metadata_dict["attributes"].items():
                stack_group.attrs[key] = value
    
    elapsed = time.time() - start_time
    logging.info(f"[purezarr]{' [SLURM '+slurm_id+']' if slurm_id else ''} "
                f"Completed write in {elapsed:.2f}s at s3://{bucket_name}/{dataset_path}")
    logging.info(f"[purezarr] Written {len(written_arrays)} pyramid levels")
