# Video Filtering with Metadata

## Description

We utilize metadata provided by YouTube to filter out videos with potentially low quality or low audio-visual correspondence.

We use the following features:
- Video Length
- Title / Description
- YouTube Category

## Command

1. Install the python package along with the dependencies

`pip install ./code/acav_metadata_filter-0.1.0-py3-none-any.whl`

2. Run the commandline interface.

`metadata_filter <path to the metadata tsv file> <path to the output tsv file>`
