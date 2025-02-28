"""Module to handle zeiss data compression"""

import logging
import os
import shutil
import sys
from pathlib import Path
from time import time
from typing import Any, List, Optional

import bioio_czi
from aind_data_transformation.core import GenericEtl, JobResponse, get_parser
from bioio import BioImage
from numcodecs.blosc import Blosc

from aind_protein_data_transformation.compress.czi_to_zarr import (
    czi_stack_zarr_writer,
)
from aind_protein_data_transformation.models import (
    CompressorName,
    SmartspimJobSettings,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
