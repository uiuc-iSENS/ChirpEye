# ChirpEye

## Introduction

## Usage

After cloning the repository, build and run the docker environment:

```sh
docker build -t chirpeye .
docker run -it --rm -p 8888:8888 chirpeye
```

Then copy the URL in the form of `http://127.0.0.1:8888/tree?token=xxx` in the terminal output and open in a browser.

## Radar Detection, Radar Parameter Estimation, and Radar Positioning Pipeline
To run the minimum working example for the ChirpEye System:

Run `pipeline.ipynb` jupyter notebook file:

The file contains tag structure parameters: 
```python
## Delay Line Parameters:
delta_L1 = (72 - 3) * 0.0254
delta_L2 = (72 - 48) * 0.0254
delta_L3 = (48 - 3) * 0.0254
expected_ratios = [
    delta_L1 / delta_L3,  # Ratio 1 (L1 to L3)
    delta_L3 / delta_L2,   # Ratio 2 (L3 to L2)
    delta_L1 / delta_L2,   # Ratio 2 (L3 to L2)
]
```

- Input: one AoA: -10 degree, slope parameter: 6666666666666 signal file, and one no radar signal file. 

1. **Radar Detection**

Important function

```python
radar_detected, snr = calculate_detection_rate_with_detectable_snr(data_dict, tolerance, threshold, "distance", range_bins, expected_ratios, return_snr=True)
```

Expected Output

```python
# For radar exists case:
radar_detected == [1.0]
Radar is detected!!!
# For no radar case:
Error for 0-0-1-0-0-0.0-0: Score too low, signal not periodic!
Radar Not Detected!!
```

2. **Radar Slope and Radar Angle of Arrival (AoA) Estimation**

Important function

```python
# Calculate and find three IF frequency peaks
_, _, valid_freq_list = find_beat_freq_triplet_with_expected_ratio(expected_ratios, tolerance, data_obj, radar_detection_mode = False, plot = False, return_snr = False, crop_to_list = True, fix_amplitude_order=True, noise_count = 10, noise_level = 7, window_length = 15)
# Calculate the estimated slope and predicted angles using three IF frequencies
predicted_slope, predicted_angle = calculate_slope_and_angle(predicted_peaks, delta_L1, delta_L2, delta_L3, d, c, speed_ratio, frequency_offsets = [0, 1300, 0])
```

Expected Output

```python
Predicted Slope: 6852467724883.765, Predicted Angle: -9.990164771550713
GT Slope: 6666666666666.667, GT Angle: -10
```

## Experiments and Naming

All Jupyter notebooks are named by corresponding figure numbers and content in the paper.

`Data/` contains all datasets we collected, with each file following `<start_frequency>-<end_frequency>-<chirp_duration>-<interchirp_dwell>-<angle_of_arrival>-<range>-<file_index>.csv` naming, where:

- `start_frequency`: Minimum frequency of the hidden radar's band, in MHz.
- `end_frequency`: Maximum frequency of the hidden radar's band, in MHz.
- `chirp_duration`: Chirp duration of the hidden radar's signal, in us.
- `interchirp_dwell`: Inter-chirp dwell duration between adjacent chirps, in us.
- `angle_of_arrival`: Angle of arrival from the hidden radar to the tag's antenna array, in degrees.
- `range`: Range from the radar to the tag, in meters.
- `file_index`: If multiple files are collected in the same configuration, different indices are given.



For other essential functions, please refer to `utils.py` for detailed documentation. 
