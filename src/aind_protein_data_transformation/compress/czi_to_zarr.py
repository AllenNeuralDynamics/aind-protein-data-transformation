"""
CZI to Zarr writer. It takes an input path
where 3D stacks are located, then these
stacks are loaded into memory and written
to zarr.
"""

import logging
import os
import time
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union, cast

import dask
import dask.array as da
import numpy as np
import xarray_multiscale
import zarr
from numcodecs import blosc
# from ome_zarr.format import CurrentFormat
# from ome_zarr.io import parse_url
# from ome_zarr.writer import write_multiscales_metadata

from aind_protein_data_transformation.compress.zarr_writer import (
    BlockedArrayWriter,
)


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [
            f"Channel:{image_name}:{i}" for i in range(data_shape[1])
        ]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    if channel_startend is None:
        channel_startend = channel_minmax

    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": "0.4",  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": data_shape[2] // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
    }
    return omero


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translation: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """
    Generate the list of coordinate transformations
    and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each
    chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset
    in physical units in each dimension

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            }
        ]
    ]
    if translation is not None:
        transforms[0].append(
            {"type": "translation", "translation": translation}
        )
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(np.ceil(lastz / scale_factor[0]))
            lasty = int(np.ceil(lasty / scale_factor[1]))
            lastx = int(np.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(
    time_unit: str = "millisecond", space_unit: str = "micrometer"
) -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr: da.Array,
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr : array-like
        The input array.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    ome_json = _build_ome(
        arr.shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()
    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, arr.chunksize, arr.shape, None
    )
    fmt.validate_coordinate_transformations(
        arr.ndim, n_lvls, coordinate_transformations
    )
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # Writing the multiscale metadata
    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)


def create_czi_opts(codec: str, compression_level: int) -> dict:
    """
    Creates CZI options for writing
    the OMEZarr.

    Parameters
    ----------
    codec: str
        Image codec used to write the image

    compression_level: int
        Compression level for the image

    Returns
    -------
    dict
        Dictionary with the blosc compression
        to write the CZI image
    """
    return {
        "compressor": blosc.Blosc(
            cname=codec, clevel=compression_level, shuffle=blosc.SHUFFLE
        )
    }


def _get_pyramid_metadata():
    """
    Gets the image pyramid metadata
    using xarray_multiscale package
    """
    return {
        "metadata": {
            "description": "Downscaling using the windowed mean",
            "method": "xarray_multiscale.reducers.windowed_mean",
            "version": str(xarray_multiscale.__version__),
            "args": "[false]",
            # No extra parameters were used different
            # from the orig. array and scales
            "kwargs": {},
        }
    }


def compute_pyramid(
    data: dask.array.core.Array,
    n_lvls: int,
    scale_axis: Tuple[int],
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
) -> List[dask.array.core.Array]:
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    metadata = _get_pyramid_metadata()

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mean,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid], metadata


def pad_array_n_d(arr, dim: int = 5):
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr

def get_shard_and_chunk_size(image_shape, shard_size, chunksize):
    
    shard_dims = tuple(min(dim, shard_size[i]) for i, dim in enumerate(image_shape))
    chunk_dims = tuple(min(dim, chunksize[i]) for i, dim in enumerate(image_shape))

    shard_dims = tuple(
        (shard // chunk) * chunk if shard % chunk != 0 else shard
        for shard, chunk in zip(shard_dims, chunk_dims)
    )

    return shard_dims, chunk_dims

def czi_stack_zarr_writer(
    image_data,
    output_path,
    voxel_size: List[float],
    final_chunksize: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    logger: logging.Logger,
    stack_name: str,
    writing_options,
    shard_shape
):
    """
    Writes a fused Zeiss channel in OMEZarr
    format. This channel was read as a lazy array.

    Parameters
    ----------
    image_data: ArrayLike
        Lazy readed CZI stack data

    output_path: PathLike
        Path where we want to write the OMEZarr
        channel

    voxel_size: List[float]
        Voxel size representing the dataset

    final_chunksize: List[int]
        Final chunksize we want to use to write
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

    """

    shard_shape = tuple([1] * (5 - len(shard_shape))) + shard_shape
    final_chunksize = tuple([1] * (5 - len(final_chunksize))) + tuple(final_chunksize)
    image_data = pad_array_n_d(arr=image_data)

    shard_shape, final_chunksize  = get_shard_and_chunk_size(
        image_data.shape,
        shard_size=shard_shape,
        chunksize=final_chunksize
    )

    # Getting channel color
    channel_colors = None

    # Rechunking dask array
    image_data = image_data.rechunk(final_chunksize)

    image_name = stack_name

    print(f"Writing {image_data} from {stack_name} to {output_path}")

    # Creating Zarr dataset
    # store = parse_url(path=output_path, mode="w").store
    root_group = zarr.group(output_path)

    # Using 1 thread since is in single machine.
    # Avoiding the use of multithreaded due to GIL

    if np.issubdtype(image_data.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(image_data.dtype).min,
            np_info_func(image_data.dtype).max,
        )
        for _ in range(image_data.shape[1])
    ]

    # Setting values for CZI
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 550.0) for _ in range(image_data.shape[1])]

    new_channel_group = root_group.create_group(
        name=image_name, overwrite=True
    )

    # Writing OME-NGFF metadata
    # write_ome_ngff_metadata(
    #     group=new_channel_group,
    #     arr=image_data,
    #     image_name=image_name,
    #     n_lvls=n_lvls,
    #     scale_factors=scale_factor,
    #     voxel_size=voxel_size,
    #     channel_names=[channel_name],
    #     channel_colors=channel_colors,
    #     channel_minmax=channel_minmax,
    #     channel_startend=channel_startend,
    #     metadata=_get_pyramid_metadata(),
    # )

    # performance_report_path = f"{output_path}/report_{stack_name}.html"

    start_time = time.time()
    # Writing zarr and performance report
    # with performance_report(filename=performance_report_path):
    logger.info(f"Writing channel {channel_name}/{stack_name}")

    # Writing zarr
    block_shape = list(
        BlockedArrayWriter.get_block_shape(
            arr=image_data, target_size_mb=12800  # 51200,
        )
    )

    # Formatting to 5D block shape
    block_shape = ([1] * (5 - len(block_shape))) + block_shape
    written_pyramid = []
    pyramid_group = None

    # Writing multiple levels
    for level in range(n_lvls):
        if not level:
            array_to_write = image_data

        else:
            # It's faster to write the scale and then read it back
            # to compute the next scale
            previous_scale = da.from_zarr(pyramid_group, pyramid_group.chunks)
            new_scale_factor = (
                [1] * (len(previous_scale.shape) - len(scale_factor))
            ) + scale_factor

            previous_scale_pyramid, _ = compute_pyramid(
                data=previous_scale,
                scale_axis=new_scale_factor,
                chunks=image_data.chunksize,
                n_lvls=2,
            )
            array_to_write = previous_scale_pyramid[-1]

        logger.info(f"[level {level}]: pyramid level: {array_to_write}")

        # Create the scale dataset

        shard_shape, final_chunksize  = get_shard_and_chunk_size(
            array_to_write.shape,
            shard_size=shard_shape,
            chunksize=array_to_write.chunksize
        )

        print(array_to_write.shape, array_to_write.chunksize, shard_shape)

        pyramid_group = new_channel_group.create_array(
            name=f"{level}",
            shape=array_to_write.shape,
            chunks=final_chunksize,
            shards=shard_shape,
            dtype=array_to_write.dtype,
            compressors=zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=3,
                shuffle=zarr.codecs.BloscShuffle.shuffle
            ),
            chunk_key_encoding={"name": "default", "separator": "/"},
            # zarr_format=3,
            overwrite=True,
        )
        
        # Block Zarr Writer
        BlockedArrayWriter.store(array_to_write, pyramid_group, block_shape)
        written_pyramid.append(array_to_write)

    end_time = time.time()
    logger.info(f"Time to write the dataset: {end_time - start_time}")
    print(f"Time to write the dataset: {end_time - start_time}")
    logger.info(f"Written pyramid: {written_pyramid}")


def example():
    from pathlib import Path

    import bioio_czi
    import dask.array as da
    from bioio import BioImage

    czi_test_stack = Path(
        "/Users/camilo.laiton/repositories/Protein/Gel1_Z2_CRTX_L3_cellanddendrites2_z-stack.czi"
    )

    if czi_test_stack.exists():
        czi_file_reader = BioImage(
            str(czi_test_stack), reader=bioio_czi.Reader
        )

        print("Dask data: ", czi_file_reader.dask_data)
        print("shape: ", czi_file_reader.shape)
        print("dims: ", czi_file_reader.dims)
        print("metadata: ", czi_file_reader.metadata)
        print("channel_names: ", czi_file_reader.channel_names)
        print("physical_pixel_sizes: ", czi_file_reader.physical_pixel_sizes)

        writing_opts = create_czi_opts(codec="zstd", compression_level=3)

        lazy_data = da.squeeze(czi_file_reader.dask_data)
        # for channel_name in
        for i, chn_name in enumerate(czi_file_reader.channel_names):
            
            czi_stack_zarr_writer(
                image_data=lazy_data[i],
                output_path=f"./{czi_test_stack.stem}",
                voxel_size=list(czi_file_reader.physical_pixel_sizes),
                final_chunksize=[128, 128, 128],
                scale_factor=[2, 2, 2],
                n_lvls=4,
                channel_name=chn_name,
                logger=logging.Logger(name="test"),
                stack_name=f"{chn_name}.zarr",
                writing_options=writing_opts["compressor"],
                shard_shape=(512, 512, 512)
            )

    else:
        print(f"File does not exist: {czi_test_stack}")


if __name__ == "__main__":
    example()
