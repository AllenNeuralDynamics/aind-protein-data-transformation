"""
CZI to Zarr S3 writer. It takes an input path
where 3D stacks are located, then these
stacks are loaded into memory and written
to zarr.
"""

import logging
import time
from typing import List, Optional

import czifile
import numpy as np
import tensorstore as ts

from aind_hcr_data_transformation.compress.omezarr_metadata import (
    _get_pyramid_metadata,
    write_ome_ngff_metadata,
)
from aind_hcr_data_transformation.utils.utils import (
    czi_block_generator,
    pad_array_n_d,
    write_json,
)


def create_spec(
    output_path: str,
    data_shape: list,
    data_dtype: str,
    shard_shape: list,
    chunk_shape: list,
    zyx_resolution: list,
    compressor_kwargs: dict,
    bucket_name: str = None,
    scale: str = "0",
    aws_region: str = "us-west-2",
) -> dict:
    """
    Create a TensorStore Zarr v3 specification for writing
    to an S3-backed dataset.

    Parameters
    ----------
    output_path : str
        Path inside the S3 bucket where the dataset will be stored.
    data_shape : list of int
        Shape of the full dataset in [t, c, z, y, x] order.
    data_dtype : str
        Data type of the dataset (e.g., 'uint16', 'float32').
    shard_shape : list of int
        Shape of the sharded outer chunks.
    chunk_shape : list of int
        Shape of the internal compressed chunks inside each shard.
    zyx_resolution : list
        Spatial resolution in microns for the z, y, x axes.
    compressor_kwargs: dict
        Compressor parameters for tensorstore
    scale : str, optional
        Scale level identifier (e.g., "0", "1", etc.). Default is "0".
    aws_region : str, optional
        AWS region where the S3 bucket resides. Default is "us-west-2".
    bucket_name : str, optional
        Name of the S3 bucket.

    Returns
    -------
    spec : dict
        TensorStore specification dictionary to create a new Zarr v3 dataset.
    """
    output_path = str(output_path)
    zyx_resolution = [
        f"{r}um" if r is not None else None for r in zyx_resolution
    ]
    zyx_resolution = [None] * (5 - len(zyx_resolution)) + zyx_resolution

    return {
        "driver": "zarr3",
        "kvstore": {
            "driver": "s3",
            "bucket": bucket_name,
            "aws_region": aws_region,
            "path": output_path,
            "aws_credentials": {"type": "default"},
        },
        "path": str(scale),
        "recheck_cached_metadata": False,
        "recheck_cached_data": False,
        "metadata": {
            "shape": data_shape,
            "zarr_format": 3,
            "node_type": "array",
            "fill_value": 0,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": shard_shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "attributes": {
                "dimension_units": zyx_resolution,
            },
            "dimension_names": ["t", "c", "z", "y", "x"],
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {
                                "name": "blosc",
                                "configuration": compressor_kwargs,
                            },
                        ],
                        "index_codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {"name": "crc32c"},
                        ],
                        "index_location": "end",
                    },
                }
            ],
            "data_type": data_dtype,
        },
        "create": True,
        "delete_existing": True,
    }


def create_downsample_dataset(
    dataset_path: str,
    start_scale: int,
    downsample_factor: list,
    downsample_mode: str,
    compressor_kwargs: dict,
    bucket_name: str,
    aws_region: str = "us-west-2",
):
    """
    Create a new downsampled scale level in a multi-scale Zarr v3 dataset.

    Parameters
    ----------
    dataset_path : str
        Base S3 path of the dataset inside the bucket.
    start_scale : int
        The scale level to downsample from (e.g., 0 for original resolution).
    downsample_factor : list of int
        Downsampling factor for each of [t, c, z, y, x] dimensions.
    downsample_mode : str
        Downsampling method. Options are: stride, median, mode, mean, min, max.
    compressor_kwargs: Dict
        Blosc compressor arguments for tensorstore
    cpu_cnt : int, optional
        Number of threads to use. If None, uses all available CPUs.
    aws_region : str, optional
        AWS region of the S3 bucket.
    bucket_name : str, optional
        S3 bucket name.
    read_cache_bytes : int, optional
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    None
        The function writes a new downsampled scale directly
        to the same Zarr dataset.
    """
    dataset_path = str(dataset_path)
    downsample_factor = [1] * (5 - len(downsample_factor)) + downsample_factor

    source_w_down_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factor,
        "downsample_method": downsample_mode,
        "base": {
            "driver": "zarr3",
            "kvstore": {
                "driver": "s3",
                "bucket": bucket_name,
                "aws_region": aws_region,
                "path": dataset_path,
            },
            "path": str(start_scale),
            "recheck_cached_metadata": False,
            "recheck_cached_data": False,
        },
    }

    downsampled_dataset = ts.open(spec=source_w_down_spec).result()
    source_dataset = downsampled_dataset.base
    new_scale = start_scale + 1

    downsampled_resolution = [
        unit.multiplier if isinstance(unit, ts.Unit) else unit
        for unit in downsampled_dataset.dimension_units
    ]

    # Creates downsample spec
    # Keeping chunk_size equal in all dimensions, however it might be
    # suboptimal? We might want to have chunks of ~50-100 MB
    down_spec = create_spec(
        output_path=dataset_path,
        data_shape=downsampled_dataset.shape,
        data_dtype=source_dataset.dtype.name,
        shard_shape=source_dataset.chunk_layout.write_chunk.shape,
        chunk_shape=source_dataset.chunk_layout.read_chunk.shape,
        zyx_resolution=downsampled_resolution,
        scale=str(new_scale),
        aws_region=aws_region,
        bucket_name=bucket_name,
        compressor_kwargs=compressor_kwargs,
    )

    down_dataset = ts.open(down_spec).result()
    try:
        with ts.Transaction() as transaction:
            downsampled_data = downsampled_dataset.with_transaction(
                transaction
            ).read().result()
    except Exception as e:
        logging.error(f"Failed to read downsampled data: {e}")
        raise e
    try:
        with ts.Transaction() as transaction:
            down_dataset.with_transaction(transaction).write(
                downsampled_data
            ).result()
        logging.info(f"Completed downsample scale {new_scale} write")
    except Exception as e:
        logging.error(f"Failed to write downsample scale {new_scale}: {e}")
        raise e

def czi_stack_zarr_writer(
    czi_path: str,
    output_path: str,
    voxel_size: List[float],
    shard_size: List[int],
    chunk_size: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    stack_name: str,
    compressor_kwargs: dict,
    bucket_name: str,
    downsample_mode: Optional[str] = "mean",
):
    """
    Writes a fused Zeiss channel in OMEZarr
    format. This channel was read as a lazy array.

    Parameters
    ----------
    czi_path: str
        Path where the CZI file is stored.

    output_path: PathLike
        Path where we want to write the OMEZarr
        channel

    voxel_size: List[float]
        Voxel size representing the dataset

    chunk_size: List[int]
        Final chunk_size we want to use to write
        the final dataset

    codec: str
        Image codec for writing the Zarr

    compression_level: int
        Compression level

    scale_factor: List[int]
        Scale factor per axis. The dimensionality
        is organized as ZYX.

    n_lvls: int
        Number of levels on the pyramid (multiresolution)
        for better visualization

    channel_name: str
        Channel name we are currently writing

    logger: logging.Logger
        Logger object

    stack_name: str
        Stack name for the zarr3

    compressor_kwargs: Dict
        Blosc compressor arguments for tensorstore

    downsample_mode: str
        Downsample mode for the dataset.
        Options are: stride, median, mode, mean, min, max
        Default is mean.

    batch_size: int = 6
        Batch size for the tensorstore tasks

    bucket_name: Optional[str] = None
        Bucket name to upload the dataset.
        If it is None, then it will be stored locally.
    """
    output_path = f"{output_path}/{stack_name}"
    start_time = time.time()

    with czifile.CziFile(str(czi_path)) as czi:
        dataset_shape = tuple(i for i in czi.shape if i != 1)
        extra_axes = (1,) * (5 - len(dataset_shape))
        dataset_shape = extra_axes + dataset_shape

        shard_size = ([1] * (5 - len(shard_size))) + shard_size
        chunk_size = ([1] * (5 - len(chunk_size))) + chunk_size

        # Getting channel color
        channel_colors = None

        logging.info(
            f"Writing from {stack_name} to {output_path} bucket {bucket_name}"
        )

        if np.issubdtype(czi.dtype, np.integer):
            np_info_func = np.iinfo

        else:
            # Floating point
            np_info_func = np.finfo

        # Getting min max metadata for the dtype
        channel_minmax = [
            (
                np_info_func(czi.dtype).min,
                np_info_func(czi.dtype).max,
            )
            for _ in range(dataset_shape[1])
        ]

        # Setting values for CZI
        # Ideally we would use da.percentile(image_data, (0.1, 95))
        # However, it would take so much time and resources and it is
        # not used that much on neuroglancer
        channel_startend = [(90.0, 1200.0) for _ in range(dataset_shape[1])]

        # Writing OME-NGFF metadata
        multiscale_zarr_json = write_ome_ngff_metadata(
            arr_shape=list(dataset_shape),
            image_name=stack_name,
            n_lvls=n_lvls,
            scale_factors=tuple(scale_factor),
            voxel_size=tuple(voxel_size),
            channel_names=[channel_name],
            channel_colors=channel_colors,
            channel_minmax=channel_minmax,
            channel_startend=channel_startend,
            metadata=_get_pyramid_metadata(),
            chunk_size=chunk_size,
            origin=[0, 0, 0],  # TODO get nominal coordinates into metadata
        )

        # Full resolution spec
        spec = create_spec(
            output_path=output_path,
            bucket_name=bucket_name,
            data_shape=list(dataset_shape),
            data_dtype=czi.dtype.name,
            shard_shape=shard_size,
            chunk_shape=chunk_size,
            zyx_resolution=voxel_size,
            compressor_kwargs=compressor_kwargs,
        )
        dataset = ts.open(spec).result()

        # shard size must be TCZYX order
        block_count = 0
        for block, axis_area in czi_block_generator(
            czi,
            axis_jumps=shard_size[-3],
            slice_axis="z",
        ):
            region = (
                slice(None),
                slice(None),
                axis_area,
                slice(0, dataset_shape[-2]),
                slice(0, dataset_shape[-1]),
            )
            try:
                with ts.Transaction() as transaction:
                    dataset[region].with_transaction(transaction).write(
                        pad_array_n_d(block)
                    ).result()
                block_count += 1
                logging.info(
                    f"Completed block {block_count} write for z-slices {axis_area}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to write block {block_count} for z-slices {axis_area}: {e}"
                )
                raise e
            # dataset[region].write(pad_array_n_d(block)).result()

        # Waiting for the tensorstore tasks
        # asyncio.run(write_tasks(tasks, batch_size=batch_size))

        for level in range(n_lvls):
            create_downsample_dataset(
                dataset_path=output_path,
                start_scale=level,
                downsample_factor=scale_factor,
                downsample_mode=downsample_mode,
                compressor_kwargs=compressor_kwargs,
                bucket_name=bucket_name,
            )

    # Writes top level json
    write_json(
        bucket_name=bucket_name,
        output_path=output_path,
        json_data=multiscale_zarr_json,
    )

    end_time = time.time()
    logging.info(f"Time to write the dataset: {end_time - start_time}")
