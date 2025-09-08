import logging
import multiprocessing
from pathlib import Path

from aind_hcr_data_transformation.compress.czi_to_zarr_purezarr import (
    czi_stack_zarr_writer,
)
from aind_hcr_data_transformation.models import ZeissJobSettings

# ----------------- CONFIG SECTION (edit as needed) -----------------
SPIM_DIR = Path(
    "/allen/aind/scratch/carson.berry/test_data/multitile_dataset/HCR_747107-14_2025-02-28_13-00-00/SPIM"
)
# OUTPUT_DIR = Path("/allen/aind/scratch/carson.berry/test_output/local_debug")
OUTPUT_DIR = Path("s3://aind-scratch-data/carson.berry/test_data_zarr3/")
MAX_CZIS = 1  # number of .czi files to process
VOXEL_SIZE_ZYX = [1.0, 1.0, 1.0]  # microns (Z,Y,X)
OVERRIDE_SHARD_SIZE_ZYX = None  # e.g. [512, 512, 512] or None to use defaults
OVERRIDE_CHUNK_SIZE_ZYX = None  # e.g. [128, 128, 128] or None
OVERRIDE_SCALE_FACTOR_ZYX = None  # e.g. [2, 2, 2] or None
OVERRIDE_LEVELS = None  # int or None
BUCKET_NAME = "aind-scratch-data"
SKIP_DOWNSAMPLE = False
LOG_LEVEL = "INFO"
# -------------------------------------------------------------------


def run_debug():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("local_debug")

    if not SPIM_DIR.is_dir():
        raise FileNotFoundError(f"SPIM dir not found: {SPIM_DIR}")

    czi_paths = sorted(SPIM_DIR.glob("*.czi"))
    if not czi_paths:
        raise FileNotFoundError(f"No .czi files in {SPIM_DIR}")
    if MAX_CZIS > 0:
        czi_paths = czi_paths[:MAX_CZIS]

    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    defaults = ZeissJobSettings.model_construct(
        input_source=SPIM_DIR.parent,
        output_directory=str(OUTPUT_DIR),
        num_of_partitions=1,
        partition_to_process=0,
    )

    shard_size = OVERRIDE_SHARD_SIZE_ZYX or defaults.shard_size
    chunk_size = OVERRIDE_CHUNK_SIZE_ZYX or defaults.chunk_size
    scale_factor = OVERRIDE_SCALE_FACTOR_ZYX or defaults.scale_factor
    n_lvls = (
        1
        if SKIP_DOWNSAMPLE
        else (OVERRIDE_LEVELS or defaults.downsample_levels)
    )

    log.info(
        f"Config shard_size={shard_size} chunk_size={chunk_size} "
        f"scale_factor={scale_factor} n_lvls={n_lvls}"
    )
    log.info(f"Selected {len(czi_paths)} CZI file(s)")

    for czi in czi_paths:
        stack_name = f"{czi.stem}.ome.zarr"
        log.info(f"Processing {czi.name} -> {stack_name}")
        czi_stack_zarr_writer(
            czi_path=str(czi),
            output_path=str(OUTPUT_DIR),
            voxel_size=VOXEL_SIZE_ZYX,
            shard_size=shard_size,
            chunk_size=chunk_size,
            scale_factor=scale_factor,
            n_lvls=n_lvls,
            channel_name=czi.stem,
            stack_name=stack_name,
            compressor_kwargs=defaults.compressor_kwargs,
            bucket_name=BUCKET_NAME,
            downsample_mode=defaults.downsample_mode,
        )
        log.info(f"Finished {czi.name}")

    log.info("All done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_debug()
