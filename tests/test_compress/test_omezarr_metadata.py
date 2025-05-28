"""
Unit tests for the omezarr_metadata module.
"""

import unittest

from numpy.testing import assert_allclose
from ome_zarr.format import CurrentFormat, FormatV01

from aind_hcr_data_transformation.compress.omezarr_metadata import (
    _build_ome,
    _compute_scales,
    _downscale_origin,
    _get_axes_5d,
    _get_pyramid_metadata,
    _validate_axes_for_format,
    _validate_omero_metadata,
    add_multiscales_metadata,
    write_ome_ngff_metadata,
)


class TestMetadataUtils(unittest.TestCase):
    """Unit tests for the metadata utility functions."""

    def test_get_pyramid_metadata(self):
        """Test that the pyramid metadata is correctly constructed."""
        result = _get_pyramid_metadata()
        self.assertIn("metadata", result)
        self.assertEqual(
            result["metadata"]["method"], "tensorstore.downsample"
        )

    def test_build_ome_defaults(self):
        """Test that the OME metadata is built with default values."""
        shape = (1, 2, 3, 4, 5)
        ome = _build_ome(shape, "test_image")
        self.assertEqual(len(ome["channels"]), 2)
        for ch in ome["channels"]:
            self.assertTrue(ch["active"])
            self.assertIn("window", ch)

    def test_compute_scales_basic(self):
        """Test that scales are computed correctly for a basic case."""
        transforms, chunks = _compute_scales(
            scale_num_levels=3,
            scale_factor=(2, 2, 2),
            pixelsizes=(1.0, 1.0, 1.0),
            chunks=(1, 1, 16, 64, 64),
            data_shape=(1, 1, 64, 256, 256),
            translations=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        )
        self.assertEqual(len(transforms), 3)
        self.assertEqual(len(chunks), 3)
        for t in transforms:
            self.assertIsInstance(t[0]["scale"], list)

    def test_get_axes_5d(self):
        """Test that the 5D axes are correctly defined."""
        axes = _get_axes_5d()
        names = [a["name"] for a in axes]
        self.assertEqual(names, ["t", "c", "z", "y", "x"])
        self.assertEqual(axes[0]["unit"], "millisecond")

    def test_validate_axes_for_format(self):
        """Test that axes are validated correctly for the current format."""
        fmt = CurrentFormat()
        axes = _get_axes_5d()
        validated_axes, ndim = _validate_axes_for_format(axes, fmt)
        self.assertEqual(ndim, 5)
        self.assertIsInstance(validated_axes, list)

    def test_validate_axes_for_format_v1(self):
        """Test that axes are validated correctly for the format v01."""
        fmt = FormatV01()
        axes = _get_axes_5d()
        validated_axes, ndim = _validate_axes_for_format(axes, fmt)
        self.assertEqual(ndim, -1)
        self.assertIsInstance(
            validated_axes, None.__class__
        )  # FormatV01 does not validate axes, so should return None

    def test_validate_omero_metadata_valid(self):
        """Test that valid OME metadata passes validation."""
        meta = _build_ome((1, 2, 3, 4, 5), "test")
        try:
            _validate_omero_metadata(meta)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_validate_omero_metadata_empty(self):
        """Test that invalid color in channel metadata raises an error."""
        self.assertEqual(_validate_omero_metadata({}), None)

    def test_validate_omero_metadata_invalid_color(self):
        """Test that invalid color in channel metadata raises an error."""
        meta = _build_ome((1, 1, 3, 4, 5), "test")
        meta["channels"][0]["color"] = "INVALID"
        with self.assertRaises(TypeError):
            _validate_omero_metadata(meta)

    def test_validate_omero_metadata_missing_window_key(self):
        """Test that missing window key in channel metadata raises an error."""
        meta = _build_ome((1, 1, 3, 4, 5), "test")
        del meta["channels"][0]["window"]["min"]
        with self.assertRaises(KeyError):
            _validate_omero_metadata(meta)

    def test_add_multiscales_metadata(self):
        """Test that multiscales metadata is added correctly."""
        group = {"name": "img"}
        datasets = [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 1, 1, 1]}
                ],
            }
        ]
        axes = _get_axes_5d()
        omero = _build_ome((1, 1, 1, 1, 1), "img")
        out = add_multiscales_metadata(
            group, datasets, axes=axes, omero_metadata=omero
        )
        self.assertIn("attributes", out)
        self.assertIn("ome", out["attributes"])
        self.assertIn("multiscales", out["attributes"]["ome"])

    def test_downscale_origin_basic(self):
        """Test that the origin is downscaled correctly."""
        origins = _downscale_origin(
            array_shape=[1, 1, 64, 64, 64],
            origin=[0.0, 0.0, 0.0],
            voxel_size=[1.0, 1.0, 1.0],
            scale_factors=[2, 2, 2],
            n_levels=3,
        )
        self.assertEqual(len(origins), 3)
        assert_allclose(origins[1][-3:], [0.5, 0.5, 0.5])


class TestWriteOMENGFFMetadata(unittest.TestCase):
    """Unit tests for the write_ome_ngff_metadata function."""

    def setUp(self):
        """Set up basic parameters for the tests."""
        self.arr_shape = [1, 2, 8, 64, 64]
        self.chunk_size = [1, 1, 4, 32, 32]
        self.image_name = "test_img"
        self.n_lvls = 2
        self.scale_factors = (2, 2, 2)
        self.voxel_size = (1.0, 1.0, 1.0)

    def test_basic_metadata(self):
        """Test that basic metadata is written correctly."""
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
        )
        self.assertIn("attributes", metadata)
        self.assertIn("ome", metadata["attributes"])
        self.assertIn("multiscales", metadata["attributes"]["ome"])
        self.assertEqual(metadata["zarr_format"], 3)

    def test_channel_metadata(self):
        """Test that channel metadata is written correctly."""
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            channel_names=["DAPI", "GFP"],
            channel_colors=None,
            channel_minmax=[(0, 255), (0, 255)],
            channel_startend=[(5, 200), (10, 240)],
        )
        channels = metadata["attributes"]["ome"]["omero"]["channels"]
        self.assertEqual(len(channels), 2)
        self.assertEqual(channels[0]["label"], "DAPI")
        self.assertEqual(channels[0]["window"]["start"], 5)
        self.assertEqual(channels[1]["window"]["end"], 240)

    def test_with_origin(self):
        """Test that metadata with origin is written correctly."""
        origin = [0.0, 0.0, 0.0]
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            origin=origin,
        )
        datasets = metadata["attributes"]["ome"]["multiscales"][0]["datasets"]
        self.assertIn("coordinateTransformations", datasets[0])
        self.assertEqual(len(datasets), self.n_lvls)

    def test_extra_metadata(self):
        """Test that extra metadata can be added."""
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            metadata={"description": "Extra meta"},
        )
        self.assertIn(
            "description", metadata["attributes"]["ome"]["multiscales"][0]
        )


if __name__ == "__main__":
    unittest.main()
