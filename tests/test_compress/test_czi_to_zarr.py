"""Test suite for the ZarrV3 data transformation module."""

import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np

from aind_hcr_data_transformation.compress.czi_to_zarr_s3 import (
    create_downsample_dataset,
    create_spec,
    czi_stack_zarr_writer,
)


class TestCreateSpec(unittest.TestCase):
    """Test cases for the create_spec function."""

    def test_create_spec_basic_parameters(self):
        """Test create_spec with basic parameters."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[1.0, 0.5, 0.5],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
            bucket_name=None,  # Add this parameter
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertEqual(spec["kvstore"]["driver"], "s3")
        self.assertEqual(spec["kvstore"]["path"], "/test/path")
        self.assertEqual(spec["path"], "0")  # default scale
        self.assertEqual(spec["metadata"]["shape"], [1, 1, 100, 200, 300])
        self.assertEqual(spec["metadata"]["data_type"], "uint16")
        self.assertEqual(
            spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"],
            [1, 1, 50, 100, 150],
        )

        # Check sharding codec configuration
        sharding_codec = spec["metadata"]["codecs"][0]
        self.assertEqual(sharding_codec["name"], "sharding_indexed")
        self.assertEqual(
            sharding_codec["configuration"]["chunk_shape"], [1, 1, 25, 50, 75]
        )

        # Check compressor is blosc with provided kwargs
        blosc_codec = sharding_codec["configuration"]["codecs"][1]
        self.assertEqual(blosc_codec["name"], "blosc")
        self.assertEqual(
            blosc_codec["configuration"], {"cname": "zstd", "clevel": 5}
        )

    def test_create_spec_with_s3_bucket(self):
        """Test create_spec with S3 bucket configuration."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="float32",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[1.0, 0.5, 0.5],
            compressor_kwargs={"cname": "lz4", "clevel": 3},
            bucket_name="test-bucket",
            aws_region="us-east-1",
        )

        self.assertEqual(spec["kvstore"]["driver"], "s3")
        self.assertEqual(spec["kvstore"]["bucket"], "test-bucket")
        self.assertEqual(spec["kvstore"]["aws_region"], "us-east-1")
        self.assertEqual(spec["kvstore"]["aws_credentials"]["type"], "default")

    def test_create_spec_resolution_formatting(self):
        """Test that zyx_resolution is properly formatted with units."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[2.0, 1.0, 1.0],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
            bucket_name="test-bucket",
        )

        expected_units = [None, None, "2.0um", "1.0um", "1.0um"]
        self.assertEqual(
            spec["metadata"]["attributes"]["dimension_units"], expected_units
        )


class TestCreateDownsampleDataset(unittest.TestCase):
    """Test cases for the create_downsample_dataset function."""

    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.create_spec")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.ts.open")
    def test_create_downsample_dataset_basic(self, mock_ts_open, mock_create_spec):
        """Test basic downsampling functionality."""

        mock_source_dataset = Mock()
        mock_source_dataset.dtype.name = "uint16"
        mock_source_dataset.chunk_layout.write_chunk.shape = [1, 1, 50, 100, 150]
        mock_source_dataset.chunk_layout.read_chunk.shape = [1, 1, 25, 50, 75]

        mock_downsampled_data = np.zeros((1, 1, 50, 100, 150), dtype=np.uint16)

        # Create a mock for the downsampled dataset
        mock_downsampled_dataset = Mock()
        mock_downsampled_dataset.base = mock_source_dataset
        mock_downsampled_dataset.shape = [1, 1, 50, 100, 150]
        mock_downsampled_dataset.dimension_units = [None, None, "2.0um", "1.0um", "1.0um"]
        
        # Mock result() method
        mock_read_result = Mock()
        mock_read_result.result.return_value = mock_downsampled_data
        mock_downsampled_dataset.read.return_value = mock_read_result

        mock_output_dataset = Mock()
        mock_write_result = Mock()
        mock_write_result.result.return_value = None
        mock_transaction_dataset = Mock()
        mock_transaction_dataset.write.return_value = mock_write_result
        mock_output_dataset.with_transaction.return_value = mock_transaction_dataset

        # Mock the ts.open results
        mock_open_result1 = Mock()
        mock_open_result1.result.return_value = mock_downsampled_dataset
        mock_open_result2 = Mock() 
        mock_open_result2.result.return_value = mock_output_dataset

        mock_ts_open.side_effect = [mock_open_result1, mock_open_result2]
        mock_create_spec.return_value = {"driver": "zarr"}

        create_downsample_dataset(
            dataset_path="/test/dataset",
            start_scale=0,
            downsample_factor=[2, 2, 2],
            downsample_mode="mean",
            compressor_kwargs={"cname": "zstd", "clevel": 5},
            bucket_name="test-bucket",
        )

        self.assertEqual(mock_ts_open.call_count, 2)
        mock_create_spec.assert_called_once()


class TestCziStackZarrWriter(unittest.TestCase):
    """Test cases for the czi_stack_zarr_writer function."""

    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.write_json")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.create_downsample_dataset")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.ts.open")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.czi_block_generator")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.create_spec")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3._get_pyramid_metadata")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.write_ome_ngff_metadata")
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr_s3.czifile.CziFile")
    def test_czi_stack_zarr_writer(
        self,
        mock_czifile,
        mock_write_ome_ngff_metadata,
        mock_get_pyramid_metadata,
        mock_create_spec,
        mock_czi_block_generator,
        mock_ts_open,
        mock_create_downsample_dataset,
        mock_write_json,
    ):
        """Test the czi_stack_zarr_writer function."""

        mock_czi = MagicMock()
        mock_czi.__enter__.return_value = mock_czi
        mock_czi.__exit__.return_value = None
        mock_czi.shape = (1, 1, 10, 20, 30)
        mock_czi.dtype = np.dtype("uint16")
        mock_czifile.return_value = mock_czi

        fake_block = np.zeros((1, 1, 10, 20, 30), dtype=np.uint16)
        mock_czi_block_generator.return_value = [(fake_block, slice(0, 10))]

        mock_dataset = MagicMock()
        mock_transaction_dataset = Mock()
        mock_write_result = Mock()
        mock_write_result.result.return_value = None
        mock_transaction_dataset.write.return_value = mock_write_result
        mock_dataset.__getitem__.return_value.with_transaction.return_value = mock_transaction_dataset
        
        mock_ts_result = MagicMock()
        mock_ts_result.result.return_value = mock_dataset
        mock_ts_open.return_value = mock_ts_result

        mock_create_spec.return_value = {"mock": "spec"}
        mock_get_pyramid_metadata.return_value = {"meta": "data"}
        mock_write_ome_ngff_metadata.return_value = {"ome": "ngff"}

        # Call the function under test
        czi_stack_zarr_writer(
            czi_path="/fake/path/image.czi",
            output_path="/fake/output",
            voxel_size=[1.0, 1.0, 1.0],
            shard_size=[10, 10, 10],
            chunk_size=[5, 5, 5],
            scale_factor=[2, 2, 2],
            n_lvls=2,
            channel_name="DAPI",
            stack_name="my_stack",
            compressor_kwargs={"cname": "zstd"},
            bucket_name="test-bucket",
            downsample_mode="mean",
        )

        # Assertions
        mock_ts_open.assert_called()
        mock_create_spec.assert_called_once()
        self.assertEqual(mock_create_downsample_dataset.call_count, 2)
        mock_write_json.assert_called_once()


class TestHelpers(unittest.TestCase):
    """Test helper methods and utilities."""

    def setUp(self):
        """Set up common test data."""
        self.sample_compressor_kwargs = {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,
        }
        self.sample_data_shape = [1, 1, 100, 200, 300]
        self.sample_voxel_size = [2.0, 1.0, 1.0]

    def test_sample_data_consistency(self):
        """Test that sample data is consistent."""
        self.assertEqual(len(self.sample_data_shape), 5)
        self.assertEqual(len(self.sample_voxel_size), 3)
        self.assertIsInstance(self.sample_compressor_kwargs, dict)
        self.assertIn("cname", self.sample_compressor_kwargs)