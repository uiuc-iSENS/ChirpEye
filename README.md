# ChirpEye

## Introduction

## Usage

After cloning the repository, build and run the docker environment:

```sh
docker build -t chirpeye .
docker run -it --rm -p 8888:8888 chirpeye
```

Then copy the URL in the form of `http://127.0.0.1:8888/tree?token=xxx` in the terminal output and open in a browser.

## Naming

All Jupyter notebooks are named by corresponding figure numbers and content in the paper.

`Data/` contains all datasets we collected, with each file following `<start_frequency>-<end_frequency>-<chirp_duration>-<interchirp_dwell>-<angle_of_arrival>-<range>-<file_index>.csv` naming, where:

- `start_frequency`: Minimum frequency of the hidden radar's band, in MHz.
- `end_frequency`: Maximum frequency of the hidden radar's band, in MHz.
- `chirp_duration`: Chirp duration of the hidden radar's signal, in us.
- `interchirp_dwell`: Inter-chirp dwell duration between adjacent chirps, in us.
- `angle_of_arrival`: Angle of arrival from the hidden radar to the tag's antenna array, in degrees.
- `range`: Range from the radar to the tag, in meters.
- `file_index`: If multiple files are collected in the same configuration, different indices are given.
