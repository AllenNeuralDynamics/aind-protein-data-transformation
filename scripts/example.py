"""
Example job
"""

from aind_protein_data_transformation.models import ZeissJobSettings
from aind_protein_data_transformation.zeiss_job import ZeissCompressionJob


def main():
    """
    Example job
    """
    basic_job_settings = ZeissJobSettings(
        input_source="/allen/aind/stage/Z1/Protein_Large_488_2025-1-16-8-00-00",
        output_directory="./test_conversion",
        num_of_partitions=4,
        partition_to_process=0,
        s3_location="aind-msma-morphology-data",
    )
    basic_job = ZeissCompressionJob(job_settings=basic_job_settings)
    basic_job.run_job()


if __name__ == "__main__":
    main()
