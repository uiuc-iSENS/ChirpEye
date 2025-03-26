import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from pprint import pprint
from scipy import signal
import time
import pandas as pd
import scipy
from scipy.stats import linregress
import math
import time
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.constants import c
from functools import reduce
import os
import seaborn as sns

speed_ratio = 0.7453    # ratio of signal speed in coax cable to c0
d = 0.1                 # antenna spacing
nfft = 2**14            # number of fft bins
l_freq = 0              # lower frequency limit
r_freq = 8e4            # upper frequency limit

def zoom_fft(x, fs, min_freq = 0, max_freq=None, nfft=2**10):
    """
    Take zoom fft in the given frequency range.

    Args:
        x (np.ndarray): input signal
        fs (float): sampling frequency
        min_freq (float): min frequency to perform fft
        max_freq (float): max frequency to perform fft
        nfft (int): number of fft bins

    Returns:
        (w, H) (np.ndarray, np.ndarray): frequency bins and fft result spectrum
    """
    if max_freq == None:
        max_freq = int(fs/2)
    
    H = signal.zoom_fft(x, fn=[min_freq,max_freq], m=nfft, fs=fs)
    w = np.linspace(min_freq, max_freq, nfft)

    return w, H

def get_output_theory(length_diff, band_width, chirp_duration, speed_ratio=0.7):
    """
    Get output beat frequency based on the given parameters.

    Args:
        length_diff (int): transmission line difference
        band_width (int): bandwidth of the chirp signal
        chirp_duration (int): chirp duration of the chirp signal
        speed_ratio (float): speed of signal ratio with respect to speed of light, default to 0.7

    Returns:
        f_beat (float): output beat frequency
    """
    inch2meter = 0.0254

    slope = band_width / chirp_duration

    length = length_diff*inch2meter 
    speed_signal = c*speed_ratio

    control_delay = length / speed_signal
    f_beat = control_delay * slope

    return f_beat


def save_numpy_with_timestamp(array, file_prefix='test'):
    """
    Save a NumPy array to a file with the current date and time as part of the filename.
    
    Parameters:
        array (numpy.ndarray): The NumPy array to be saved.
        file_prefix (str): Prefix for the filename (optional).
        
    Returns:
        str: The filename where the array is saved.
    """
    # Generate current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create filename with prefix and timestamp
    filename = f"{file_prefix}_{current_datetime}.npy"
    
    # Save array to file
    np.save(filename, array)
    
    return filename

def calculate_expected_period(amplitude, fs, plot=False):
    """
    Calculate the expected period of the signal.

    Args:
        amplitude (np.ndarray): The input signal.
        fs (float): The sampling frequency of the signal.
        plot (bool, optional): Whether to plot the debug spectrum and print debug information.

    Returns:
        expected_period (float): The expected period of the signal.
    """

    amplitude -= np.mean(amplitude)
    fft_spectrum = np.abs(scipy.fft.rfft(amplitude, n = nfft))
    fft_freq = scipy.fft.rfftfreq(nfft, 1/fs)

    l_chirp_freq = 1 / 1e-3
    r_chirp_freq = 1 / 150e-6
    l_interval = int(l_chirp_freq / fs * nfft)
    r_interval = int(r_chirp_freq / fs * nfft)
    max_points = int(len(fft_freq) / r_interval)

    intervals = np.arange(l_interval, r_interval)
    sum_fft = np.zeros_like(intervals, dtype = 'float')
    for interval in range(l_interval, r_interval):
        sum_fft[interval - l_interval] = np.sum(fft_spectrum[:max_points*interval:interval])

    expected_period = nfft / (fs * intervals[np.argmax(sum_fft)])
    prominence, _, _ = signal.peak_prominences(sum_fft, [np.argmax(sum_fft)])
    score = prominence[0] / np.std(sum_fft)

    # Filter only positive differences
    # positive_diffs = frequency_diffs[frequency_diffs > 0]
    if plot:
        plt.figure()
        plt.plot(1 / (intervals * fs / nfft) * 1e6, sum_fft)
        plt.xlabel('Period (us)')
        plt.ylabel('Magnitude')
        plt.title('Period estimation spectrum')
        plt.show()

        print("Expected Period: ", expected_period * 1e6, "us")
        print("Score: ", score)
    return expected_period, score

def get_seg_idx_std(amplitude, fs, expected_period = None, plot = False, plot_end_idx = 500, window_length = 20, ignore_aperiodic = False):
    """
    Calculate segmentation indices based on signal periodicity.

    Args:
        amplitude (np.ndarray): The input signal amplitude.
        fs (float): The sampling frequency of the signal.
        expected_period (float, optional): The expected period of the signal in seconds.
            If None, it will be calculated using `calculate_expected_period`.
        plot (bool, optional): Whether to plot the raw signal, correlation, and segmentation points (debug).
        plot_end_idx (int, optional): The maximum index for plotting the signal.
        window_length (int, optional): The length of the sliding window for calculating correlation.
        ignore_aperiodic (bool, optional): Whether to ignore signals that are not periodic. If False, raises an error for low periodicity.

    Returns:
        tuple:
            - peaks (np.ndarray): Indices of the segmentation points in the signal.
            - corr (np.ndarray): The normalized correlation array used for peak detection.

    Raises:
        ValueError: If the expected period score is too low and `ignore_aperiodic` is False.
    """

    if expected_period is None:
        expected_period, score = calculate_expected_period(amplitude, fs, plot = plot)
        if score < 5:
            if not ignore_aperiodic:
                raise ValueError("Score too low, signal not periodic!")
            expected_period = 300e-6

    corr = np.zeros(len(amplitude) - window_length)
    for i in range(len(amplitude) - window_length):
        corr[i] = -np.std(amplitude[i:i+window_length])
    corr = (corr - np.min(corr)) / np.std(corr)
    corr *= corr
    corr = signal.savgol_filter(corr, 20, 2)

    peaks, _ = signal.find_peaks(corr, height = 5, distance = int(expected_period * fs * 0.7))
    peaks += window_length // 2

    if plot:
        peaks_end_idx = np.argmax(peaks > plot_end_idx)
        plt.figure(figsize=(16, 9))
        plt.plot(amplitude[:plot_end_idx], label = "Raw signal")
        plt.plot(corr[:plot_end_idx] / np.std(corr), label = "Correlation")
        plt.scatter(peaks[:peaks_end_idx], amplitude[peaks[:peaks_end_idx]], c='r', marker = 'o', zorder = 3, label = "Segmentation points")
        plt.legend()
    
    return peaks, corr

# !!! Note - num_segment was changed from 3 to None recently. If old code breaks, supply num_segments = 3.
def crop_signal(amplitude, fs, expected_period = None, num_segments = None, num_skip_period = 5, crop_factor = None, plot = False, window_length = 20, ignore_aperiodic = False, plot_end_idx = 500, return_snr = False):
    """
        Crops the raw signal into the MUSIC input matrix X.

        Args:
            amplitude (np.ndarray): Raw signal.
            fs (float): Sampling frequency.
            num_segments (int, optional): Number of periods to use. If None, use all periods except first and last `num_skip_period` periods.
            num_skip_period (int, optional): Number of periods to skip at start and end of signal.
            crop_factor (int, optional): Factor to crop the signal. If None, no cropping is done.
            plot (bool, optional): Whether to plot the segmentation points (debug).
            window_length (int, optional): Length of the window to use for correlation in segmentation.
            ignore_aperiodic (bool, optional): Whether to ignore aperiodic signals and throw an error.
            plot_end_idx (int, optional): index to plot up to.
            return_snr (bool, optional): whether to return the SNR.

        Returns:
            tuple (tuple):
                - X (np.ndarray): A 2D matrix where each column is a cropped segment of the signal.
                - segment_indices (np.ndarray): Indices of segmentation points in the signal.
                - snr (float, optional): The signal-to-noise ratio, returned if `return_snr` is True.
    """
    segment_indices, _ = get_seg_idx_std(amplitude, fs, expected_period = expected_period, plot = plot, window_length = window_length, ignore_aperiodic = ignore_aperiodic, plot_end_idx=plot_end_idx)

    segment_length = np.max(np.diff(segment_indices))
    if crop_factor is not None:
        crop_start_padding = int(segment_length // crop_factor)
        segment_length = segment_length - 2 * crop_start_padding
    else:
        raise ValueError("Please specify crop_factor")
    if num_segments is None:
        num_segments = len(segment_indices) - 2 * num_skip_period - 1
    
    X = np.zeros((segment_length, num_segments))
    signal_power = 0
    interchirp_delay_power = 0
    for idx in range(num_segments):
        offsetted_idx = idx + num_skip_period
        segment = amplitude[segment_indices[offsetted_idx] + crop_start_padding : segment_indices[offsetted_idx] + crop_start_padding + segment_length]
        X[:, idx] = segment - np.mean(segment)
        # X[:, idx] = segment - segment[0]
        if return_snr:
            interchirp_power_length = int(segment_length // (crop_factor * 2))
            interchirp_signal = np.concatenate((amplitude[segment_indices[offsetted_idx] : segment_indices[offsetted_idx] + interchirp_power_length], amplitude[segment_indices[offsetted_idx + 1] - interchirp_power_length : segment_indices[offsetted_idx + 1]]))
            segment_normalized = segment - np.mean(interchirp_signal)
            interchirp_signal_normalized = interchirp_signal - np.mean(interchirp_signal)

            interchirp_delay_power += np.mean(interchirp_signal_normalized ** 2)
            signal_power += np.mean(segment_normalized ** 2)

    if return_snr:
        return X, segment_indices, signal_power / interchirp_delay_power
    return X, segment_indices

# ? TODO Check if delete
def crop_signal_tolist(amplitude, fs, expected_period=None, num_segments = 3, num_skip_period = 5, crop_factor = None, plot = False, window_length = 20, ignore_aperiodic = False):
    """
        Crops the raw signal into the MUSIC input matrix X.
        amplitude: raw signal
        fs: sampling frequency
        num_segments: number of periods to use. If none, use all periods except first and last `num_skip_period` periods.
    """
    segment_indices, _ = get_seg_idx_std(amplitude, fs, expected_period = expected_period, plot = plot, window_length = window_length, ignore_aperiodic = ignore_aperiodic)

    segment_length = np.max(np.diff(segment_indices))
    if crop_factor is not None:
        crop_start_padding = int(segment_length // crop_factor)
        segment_length = segment_length - 2 * crop_start_padding
    else:
        raise ValueError("Please specify crop_factor")
    if num_segments is None:
        num_segments = len(segment_indices) - 2 * num_skip_period - 1
        
        
    num_of_signal = (len(segment_indices) - 1) // num_segments
    
    Xs = []
    for n_signal in range(num_of_signal):
        X = np.zeros((segment_length, num_segments))
        for idx in range(num_segments):
            offsetted_idx = n_signal * idx + num_skip_period
            segment = amplitude[segment_indices[offsetted_idx] + crop_start_padding : segment_indices[offsetted_idx] + crop_start_padding + segment_length]
            X[:, idx] = segment - np.mean(segment)
        Xs.append(X)
    Xs = np.array(Xs)
    return Xs, segment_indices

# ? TODO Check if delete
def detect_frequency_classic_music(X, fs, data_dic = None, signal_count = 2, dim_signal_space = None, plot = False, l_freq = 0, r_freq = 8e4, nfft = 2**14):
    if dim_signal_space is None:
        dim_signal_space = 2 * signal_count

    U, S, Vh = np.linalg.svd(X @ X.T, full_matrices=False)
    t = np.arange(X.shape[0]) / fs
    freqs = np.linspace(l_freq, r_freq, nfft)
    steering_vecs = np.exp(-1j * 2 * np.pi * np.outer(freqs, t)) # nfft * t
    spectrum = 1 / ( np.linalg.norm(steering_vecs @ U[:, dim_signal_space:], axis = 1)**2 )

    peaks, _ = signal.find_peaks(spectrum)

    # Get the corresponding frequencies for the sorted peaks
    sorted_peak_idx = peaks[np.argsort(spectrum[peaks])][::-1][:signal_count]
    sorted_peak_freqs = freqs[sorted_peak_idx]

    if plot:
        plt.figure()
        plt.plot(freqs, spectrum)
        for i in range(signal_count):
            plt.axvline(sorted_peak_freqs[i], color = 'r', linestyle = '--', label = f"{sorted_peak_freqs[i]:.2f} Hz")
        if data_dic is not None:
            peak1_freq = data_dic["peak1"]
            peak2_freq = data_dic["peak2"]
            plt.axvline(x=peak1_freq, color='r', label=f'Peak 1: {peak1_freq} Hz')
            plt.axvline(x=peak2_freq, color='g', label=f'Peak 2: {peak2_freq} Hz')
            plt.title(f"Frequency Spectrum for {data_dic['filename']}")
        plt.legend()

    return sorted_peak_freqs, spectrum

# ? TODO Check if delete
def detect_frequency_old(X, fs, data_dic = None, signal_count = 2, dim_signal_space = None, plot = False, l_freq = 0, r_freq = 8e4, nfft = 2**14):
    """
        Classic MUSIC, taking into account both signal strength and noise perpendicularity.

        Args:
            X (np.ndarray): Regular input for MUSIC-based frequency detection, shape (num_samples, num_periods)
            fs (float): sampling frequency
            signal_count (int, optional): number of frequencies to detect. `2*signal_count` is the actual signal space dimension.
            dim_signal_space (int, optional): number of signal eigenvectors to consider. If None, set to `2*signal_count`.
            noise_count (int, optional): number of noise eigenvectors to consider. If None, set to `num_samples - signal_count`.
            plot (bool, optional): whether to plot the frequency spectrum for debug.
            l_freq (float, optional): lower frequency limit for beat frequency spectrum
            r_freq (float, optional): upper frequency limit for beat frequency spectrum
            nfft (int, optional): number of fft bins

        Returns:
            tuple (tuple):
                - sorted_peak_freqs (np.ndarray): detected frequencies
                - spectrum (np.ndarray): frequency spectrum
    """

    if dim_signal_space is None:
        dim_signal_space = 2 * signal_count

    U, S, Vh = np.linalg.svd(X @ X.T, full_matrices=False)

    t = np.arange(X.shape[0]) / fs
    freqs = np.linspace(l_freq, r_freq, nfft)

    steering_vecs = np.exp(-1j * 2 * np.pi * np.outer(freqs, t)) # nfft * t
    spectrum = 1 / ( np.linalg.norm(steering_vecs.real @ U[:, dim_signal_space:], axis = 1) + np.linalg.norm(steering_vecs.imag @ U[:, dim_signal_space:], axis = 1) )

    # ? Non-vectorized version:
    # for i in range(nfft):
    #     steering_vec = np.exp(-1j * 2 * np.pi * freqs[i] * t)
    #     spectrum[i] = 1 / (np.linalg.norm(U[:, 2 * signal_count:].T @ steering_vec.real) + np.linalg.norm(U[:, 2 * signal_count:].T @ steering_vec.imag))
        
    peaks, _ = signal.find_peaks(spectrum)

    # Get the corresponding frequencies for the sorted peaks
    sorted_peak_idx = peaks[np.argsort(spectrum[peaks])][::-1][:signal_count]
    sorted_peak_freqs = freqs[sorted_peak_idx]
    if plot:
        plt.figure()
        plt.plot(freqs, spectrum)
        for i in range(signal_count):
            plt.axvline(sorted_peak_freqs[i], color = 'r', linestyle = '--', label = f"{sorted_peak_freqs[i]:.2f} Hz")
        if data_dic is not None:
            peak1_freq = data_dic["peak1"]
            peak2_freq = data_dic["peak2"]
            plt.axvline(x=peak1_freq, color='r', label=f'Peak 1: {peak1_freq} Hz')
            plt.axvline(x=peak2_freq, color='g', label=f'Peak 2: {peak2_freq} Hz')
            plt.title(f"Frequency Spectrum for {data_dic['filename']}")
        plt.legend()

    return sorted_peak_freqs, spectrum

def detect_frequency(X, fs, data_dic = None, signal_count = 3, dim_signal_space = None, noise_count = None, plot = False, l_freq = 0, r_freq = 8e4, nfft = 2**14):
    """
        Spectra-MUSIC, taking into account both signal strength and noise perpendicularity.

        Args:
            X (np.ndarray): Regular input for MUSIC-based frequency detection, shape (num_samples, num_periods).
            fs (float): Sampling frequency.
            signal_count (int, optional): Number of frequencies to detect. `2*signal_count` is the actual signal space dimension.
            dim_signal_space (int, optional): Number of signal eigenvectors to consider. If None, set to `2*signal_count`.
            noise_count (int, optional): Number of noise eigenvectors to consider. If None, set to `num_samples - signal_count`.
            plot (bool, optional): Whether to plot the frequency spectrum for debug.
            l_freq (float, optional): Lower frequency limit for beat frequency spectrum.
            r_freq (float, optional): Upper frequency limit for beat frequency spectrum.
            nfft (int, optional): Number of fft bins.

        Returns:
            tuple (tuple):
                - sorted_peak_freqs (np.ndarray): detected frequencies
                - spectrum (np.ndarray): frequency spectrum
    """
    U, S, Vh = np.linalg.svd(X @ X.T, full_matrices=False)
    if noise_count is not None:
        dim_signal_space = X.shape[0] - noise_count
    if dim_signal_space is None:
        dim_signal_space = 2 * signal_count

    t = np.arange(X.shape[0]) / fs
    freqs = np.linspace(l_freq, r_freq, nfft)

    steering_vecs = np.exp(-1j * 2 * np.pi * np.outer(freqs, t)) # nfft * t
    noise_corrs = np.linalg.norm(steering_vecs.real @ U[:, dim_signal_space:], axis = 1) + np.linalg.norm(steering_vecs.imag @ U[:, dim_signal_space:], axis = 1) # nfft
    signal_corrs = np.zeros(nfft)
    for signal_idx in range(dim_signal_space):
        signal_corrs += S[signal_idx] * ( (steering_vecs.real @ U[:, signal_idx]) ** 2 + (steering_vecs.imag @ U[:, signal_idx]) ** 2 )
    spectrum = signal_corrs / noise_corrs

    peaks, _ = signal.find_peaks(spectrum)

    # Get the corresponding frequencies for the sorted peaks
    sorted_peak_idx = peaks[np.argsort(spectrum[peaks])][::-1][:signal_count]
    sorted_peak_freqs = freqs[sorted_peak_idx]

    if plot:
        plt.figure()
        plt.plot(freqs, spectrum / 1000)
        for i in range(signal_count):
            plt.axvline(sorted_peak_freqs[i], color = 'r', linestyle = ':', label = f"{sorted_peak_freqs[i]:.2f} Hz")
        if data_dic is not None:
            if "peak1" in data_dic:
                peak1_freq = data_dic["peak1"]
                peak2_freq = data_dic["peak2"]
                peak3_freq = data_dic["peak3"]
                plt.axvline(x=peak1_freq, color='r', label=f'GT Peak 1: {peak1_freq} Hz')
                plt.axvline(x=peak2_freq, color='g', label=f'GT Peak 2: {peak2_freq} Hz')
                plt.axvline(x=peak3_freq, color='g', label=f'GT Peak 2: {peak2_freq} Hz')
            elif "gt_freqs" in data_dic:
                for i in range(len(data_dic["gt_freqs"])):
                    plt.axvline(x=data_dic["gt_freqs"][i], color='g', label=f'GT Peak {i+1}: {data_dic["gt_freqs"][i]} Hz')
            plt.title(f"Frequency Spectrum for {data_dic['filename']}")
        plt.legend()
    return sorted_peak_freqs, spectrum

def detect_frequency_cluster(X, fs, freq_ratio, dim_signal_space = None, calibration = None, l_freq = 0, r_freq = 8e4, nfft = 2**14, plot = False):
    """
        Detects frequency clusters with known frequency ratios and amplitude ratios.
        Params:
            X: regular input for MUSIC-based frequency detection, shape (num_samples, num_periods)
            fs: sampling frequency
            freq_ratio: frequency ratios of the signals. Does not need to be normalized. Ex. [33, 36]
            amp_ratio: amplitude ratios of the signals. Does not need to be normalized. Ex. [1, 1.5]
            total_signal_count: total number of signals to detect. This includes single frequency peaks. NOT THE NUMBER OF SIGNALS IN THE CLUSTER!!!!!
    """

    freq_ratio = np.array(freq_ratio).copy() / freq_ratio[0]
    if calibration is None:
        calibration = np.zeros_like(freq_ratio)
    if dim_signal_space is None:
        dim_signal_space = len(freq_ratio) * 2
    signal_count = len(freq_ratio)

    U, S, Vh = np.linalg.svd(X @ X.T, full_matrices=False)

    noise_corrs = np.zeros(nfft)
    signal_corrs = np.ones(nfft)
    t = np.arange(X.shape[0]) / fs
    base_freqs = np.linspace(l_freq, r_freq, nfft)

    for freq_ratio_idx in range(len(freq_ratio)):
        steering_vecs = np.exp(-1j * 2 * np.pi * np.outer(base_freqs * freq_ratio[freq_ratio_idx] + calibration[freq_ratio_idx], t)) # nfft * t
        noise_corrs += np.linalg.norm(steering_vecs.real @ U[:, dim_signal_space:], axis = 1) + np.linalg.norm(steering_vecs.imag @ U[:, dim_signal_space:], axis = 1) # nfft
        single_freq_signal_corrs = np.zeros(nfft)
        for signal_idx in range(dim_signal_space):
            single_freq_signal_corrs += S[signal_idx] * ( (steering_vecs.real @ U[:, signal_idx]) ** 2 + (steering_vecs.imag @ U[:, signal_idx]) ** 2 )
        signal_corrs *= single_freq_signal_corrs
    spectrum = signal_corrs / noise_corrs

    # t = np.arange(X.shape[0]) / fs
    # freqs = np.linspace(l_freq, r_freq, nfft)
    # spectrum = np.zeros(nfft)
    # for i in range(nfft):
    #     # steering_vec = np.zeros_like(t, dtype = "complex")
    #     noise_corr = 0
    #     signal_corr = 0
    #     for j in range(signal_count):
    #         individual_steering_vec = np.exp(-1j * 2 * np.pi * (freqs[i] * freq_ratio[j] + calibration[j]) * t)
    #         # steering_vec += individual_steering_vec
    #         noise_corr += np.linalg.norm(U[:, dim_signal_space:].T @ individual_steering_vec.real) + np.linalg.norm(U[:, dim_signal_space:].T @ individual_steering_vec.imag)
    #         for signal_idx in range(dim_signal_space):
    #             signal_corr += S[signal_idx] * ( (U[:, signal_idx].T @ individual_steering_vec.real) ** 2 + (U[:, signal_idx].T @ individual_steering_vec.imag) ** 2 )
    #     # noise_corr += np.linalg.norm(U[:, dim_signal_space:].T @ steering_vec.real) + np.linalg.norm(U[:, dim_signal_space:].T @ steering_vec.imag)

    #     # print((U[:, signal_idx].T @ steering_vec.real) ** 2 + (U[:, signal_idx].T @ steering_vec.imag) ** 2)
    #     spectrum[i] = signal_corr / noise_corr

    if plot:
        plt.figure()
        plt.plot(base_freqs, spectrum)

    peaks, _ = signal.find_peaks(spectrum)

    # Get the corresponding frequencies for the sorted peaks
    sorted_peak_idx = peaks[np.argsort(spectrum[peaks])][::-1][0]
    ref_freq = base_freqs[sorted_peak_idx]
    detected_freqs = ref_freq * freq_ratio + calibration

    if plot:
        for detected_freq in detected_freqs:
            plt.axvline(detected_freq, color = 'r', linestyle = ':', label = f"{detected_freq:.2f} Hz")
        plt.legend()
    return detected_freqs, spectrum

def detect_frequency_spatial_smoothing(X, fs, data_dic = None, signal_count = 3, dim_signal_subspace = None, l_freq = 0, r_freq = 8e4, nfft = 2**14, spatial_window = 20, plot = False):
    spatial_window = 20
    windowed_X = np.zeros((spatial_window, X.shape[1] * (X.shape[0] - spatial_window + 1)))
    for i in range(X.shape[0] - spatial_window + 1):
        for j in range(X.shape[1]):
            windowed_X[:, j * (X.shape[0] - spatial_window + 1) + i] = X[i: i + spatial_window, j]

    return detect_frequency_old(windowed_X, fs, data_dic, signal_count = 3, dim_signal_space = dim_signal_subspace, plot = plot)

def butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth low-pass filter.

    Args:
        cutoff (float): The cutoff frequency of the filter in Hz.
        fs (float): The sampling frequency of the signal in Hz.
        order (int, optional): The order of the Butterworth filter. Defaults to 5.

    Returns:
        tuple (tuple):
            - b (np.ndarray): Numerator coefficients of the filter.
            - a (np.ndarray): Denominator coefficients of the filter.
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth low-pass filter to a signal.

    Args:
        data (np.ndarray): The input signal to be filtered.
        cutoff (float): The cutoff LP frequency of the filter in Hz.
        fs (float): The sampling frequency of the signal in Hz.
        order (int, optional): The order of the Butterworth filter. Defaults to 5.

    Returns:
        y (np.ndarray): The filtered signal.
    """

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def parse_csv_file(filepath):
    """
    Parse a CSV file to extract time and voltage data.

    Args:
        filepath (str): Path to the CSV file. The file should contain time and voltage data,
                        with the first 5 rows and the last 5 rows to be ignored.

    Returns:
        data (np.ndarray): A 2D array where the first column contains time values
                    and the second column contains voltage values.
    """

    df = pd.read_csv(filepath, skiprows=5, names=["Time", "Voltage"])
    df = df.iloc[:-5]
    data = np.zeros((len(df), 2))
    data[:, 0] = df["Time"].values
    data[:, 1] = df["Voltage"].values
    return data

def get_setup_angle(lengths):
    """
    Calculate delay line length difference and equivalent angle of AoA-induced delay difference between each pair of delay lines.

    Args:
        lengths (list or np.ndarray): Lengths of delay line.

    Returns:
        tuple (tuple):
            - setup_diff (np.ndarray): Differences in delay line length between all pairs of delay lines.
            - angle_diff (np.ndarray): Differences in equivalent angle of AoA-induced delay between all pairs.
    """
    setup_diff, angle_diff = [], []
    for i in range(len(lengths)):
        for j in range(i+1, len(lengths)):
            if lengths[j] > lengths[i]:
                setup_diff.append(lengths[j] - lengths[i])
                angle_diff.append(j - i)
            else:
                setup_diff.append(lengths[i] - lengths[j])
                angle_diff.append(i - j)
    return np.array(setup_diff), np.array(angle_diff)

def get_theory_freq(slope, lengths, angle, antenna_spacing = 0.1, speed_ratio=0.7453):
    """
    Calculate the theoretical beat frequencies based on setup parameters.

    Args:
        slope (float): The slope of the chirp signal in Hz/s.
        lengths (list or np.ndarray): A list or array of lengths for the setup.
        angle (float): The angle of arrival in degrees.
        antenna_spacing (float, optional): The spacing between adjacent antennas in meters. Defaults to 0.1.
        speed_ratio (float, optional): The ratio of the medium propagation speed to the speed of light in coax cable. Defaults to 0.7453.

    Returns:
        np.ndarray: The theoretical beat frequencies.
    """
    theta_delay = antenna_spacing * np.sin(np.deg2rad(angle)) / c * slope
    setup_diff, angle_diff = get_setup_angle(lengths)
    return setup_diff / (c * speed_ratio) * slope + theta_delay * angle_diff

def default_theory_freq(filename):
    """
    Calculate the theoretical beat frequencies based on dataset name for the default delay line lengths.

    Args:
        filename (str): The name of the dataset. All parameters are extracted from the filename.

    Returns:
        np.ndarray: The theoretical beat frequencies.
    """
    start_freq, stop_freq, cd_time, dwells, angle, distance, index = parse_bin_name(filename)
    if start_freq == 0:
        return np.array([0, 0, 0])
    setup = np.array([72, 3, 48]) * 0.0254
    bw = (stop_freq - start_freq) * 1e6
    cd = cd_time * 1e-6
    S = bw / cd
    return get_theory_freq(S, setup, angle)

def default_theory_freq_from_params(start_freq, stop_freq, cd_time, angle):
    """
    Calculate the theoretical beat frequencies based on parameters for the default delay line lengths.

    Args:
        start_freq (float): Start frequency in MHz.
        stop_freq (float): Stop frequency in MHz.
        cd_time (float): Chirp duration in microseconds (us).
        angle (float): Angle of arrival in degrees.

    Returns:
        np.ndarray: The theoretical beat frequencies.
    """

    if start_freq == 0:
        return np.array([0, 0, 0])
    setup = np.array([72, 3, 48]) * 0.0254
    bw = (stop_freq - start_freq) * 1e6
    cd = cd_time * 1e-6
    S = bw / cd
    return get_theory_freq(S, setup, angle)

def parse_bin_file(filepath):
    """
    Parse a binary file to extract structured data.

    Args:
        filepath (str): Path to the binary file containing comma-separated data.

    Returns:
        parsed_data (np.ndarray): Parsed data (voltage from envelope detector).
    """

    parsed_data = []

    with open(filepath, 'rb') as file:
        # Read the entire file content
        content = file.read()

        # Decode the content to a string
        content_str = content.decode('utf-8').strip()

        # Split the content by lines
        # lines = content_str.split('\r\n')
        lines = content_str.split('\n')

        for line in lines:
            if line and ',' in line and line.split(',')[1] != "":
                # Split each line by comma and convert to tuple of integers
                values = list(map(int, line.split(',')))
                parsed_data.append(values)
        file.close()

    return parsed_data

def parse_bin_name(filename):
    """
    Parse a binary filename to extract parameters.

    Args:
        filename (str): The full path or name of the binary file. The filename should follow the format:
            "<start_freq>-<stop_freq>-<cd_time>-<dwells>-<angle>-<distance>-<index>.ext".

    Returns:
        tuple: A tuple containing the following parameters:
            - start_freq (int): Start frequency in MHz.
            - stop_freq (int): Stop frequency in MHz.
            - cd_time (int): Chirp duration in microseconds (us).
            - dwells (int): Inter-chirp dwell time in microseconds (us).
            - angle (int): Angle in degrees (negative if prefixed with 'n' or 'm').
            - distance (int): Distance in meters (m).
            - index (int): Index of the file.
    """

    split = filename.split("/")[-1].split(".")
    params = split[0].split("-")
    start_freq = int(params[0])  # MHz
    stop_freq = int(params[1])  # MHz
    cd_time = int(params[2])  # us
    dwells = int(params[3])  # us
    if params[4].startswith("m") or params[4].startswith("n"):
        angle = -int(params[4][1:])
    else:
        angle = int(params[4])  # degree
    distance = int(params[5])  # m
    index = int(params[6])
    return start_freq, stop_freq, cd_time, dwells, angle, distance, index


def read_data_dict(directory_path):
    filenames = os.listdir(directory_path)
    data_dict = {}
    for filename in filenames:
        start_freq, stop_freq, cd, dwells, angle, distance, index = parse_bin_name(filename)
        gt_freqs = default_theory_freq(filename)
        bw = (stop_freq - start_freq) * 1e6
        cd = cd * 1e-6
        gt_slope = bw / cd
        
        data = parse_csv_file(os.path.join(directory_path, filename))
        key = f"{distance}-{angle}-{index}-{start_freq}-{stop_freq}-{cd}-{dwells}"
        if key not in data_dict:
            data_dict[key] = {"data": data, "gt_freqs": gt_freqs, "CD": cd, "angle": angle, "distance": distance, "bw":  bw, "slope": gt_slope, "filename": filename, "start_freq": start_freq}
    return data_dict

def default_expected_ratio():
    delta_L1 = (72 - 3) * 0.0254
    delta_L2 = (72 - 48) * 0.0254
    delta_L3 = (48 - 3) * 0.0254
    expected_ratios = [
        delta_L1 / delta_L3,  # Ratio 1 (L1 to L3)
        delta_L3 / delta_L2,   # Ratio 2 (L3 to L2)
        delta_L1 / delta_L2,   # Ratio 2 (L3 to L2)
    ]
    return expected_ratios

def find_beat_freq_triplet_with_expected_ratio(expected_ratios, tolerance, data_obj, radar_detection_mode, small_peak_cal = 400, plot = False, return_snr = False, crop_to_list = False, noise_count = None, fix_amplitude_order = False, noise_level = 10, window_length = 20):
    """
    Identifies a triplet of beat frequencies in a signal that match expected frequency ratios within a given tolerance.
    Parameters:
        expected_ratios (list of float): The expected frequency ratios to match. 
            For radar detection mode, this should contain two ratios [ratio_1, ratio_2].
            Otherwise, it should contain three ratios [ratio_1, ratio_2, ratio_3].
        tolerance (float): The allowable deviation from the expected ratios.
        data_obj (dict): A dictionary containing the signal data. 
            Must include 'data' (2D array with time and amplitude) and 'distance' (float).
        radar_detection_mode (bool): If True, uses a stricter check for radar detection.
        small_peak_cal (int, optional): Calibration value added to the smallest peak for ratio calculation. Default is 400.
        plot (bool, optional): If True, plots the frequency spectrum and detected peaks. Default is False.
        return_snr (bool, optional): If True, returns the signal-to-noise ratio (SNR). Default is False.
        crop_to_list (bool, optional): If True, crops the signal into multiple segments for analysis. Default is False.
        noise_count (int, optional): Number of noise peaks to consider for peak detection. Default is None.
        fix_amplitude_order (bool, optional): If True, ensures the amplitude of the largest peak is greater than or equal to the others. Default is False.
        noise_level (int, optional): Threshold for peak height relative to the maximum peak value. Default is 10.
        window_length (int, optional): Length of the window used for signal cropping. Default is 20.
    Returns:
        tuple: A tuple containing:
            - best_triplet (tuple or None): The indices of the three peaks that match the expected ratios, or None if no valid triplet is found.
            - snr (float): The signal-to-noise ratio, if `return_snr` is True. Otherwise, returns 0.
            - valid_freq_list (list of list of float): A list of valid frequency triplets (in Hz) that match the expected ratios.
    """
    # Convert frequency indices to actual frequencies
    l_freq = 0
    r_freq = 8e4
    nfft = 2**14
    frequency_axis = np.linspace(l_freq, r_freq, nfft)
    timestamp = data_obj['data'][:, 0]
    amplitude = data_obj['data'][:, 1] * 1.0
    fs = len(timestamp) / (timestamp[-1] - timestamp[0])
    distance = data_obj['distance']
    valid_freq_list = []
    # Crop the signal
    Xs = None
    if not crop_to_list:
        X, _, snr = crop_signal(amplitude, fs, num_segments = None, crop_factor = 8, plot = False, window_length = window_length, return_snr = return_snr)
    else:
        Xs, segment_indices = crop_signal_tolist(amplitude, fs, None, num_segments = 5, crop_factor = 8, plot = False, window_length = window_length)
        snr = 0
    if Xs is None:
        Xs = [X]
    for index, X in enumerate(Xs):
        peaks, frequency_spectrum = detect_frequency(X, fs, None, signal_count = 3, plot = plot, noise_count=noise_count, l_freq = l_freq, r_freq = r_freq, nfft = nfft)
        max_peak_value = np.max(frequency_spectrum)
        # Adjust peak height threshold based on noise level
        peaks, properties = signal.find_peaks(frequency_spectrum, height=max_peak_value/noise_level)
        
        peak_amplitudes = properties['peak_heights']
        
        best_triplet = None

        min_peak_distance = 1000
        max_amplitude_prod = 0
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                for k in range(j + 1, len(peaks)):
                    if peaks[k] > peaks[j] > peaks[i]:
                        if (peaks[j] - peaks[i] < min_peak_distance) or (peaks[k] - peaks[j] < min_peak_distance) or (peaks[k] - peaks[i] < min_peak_distance):
                            continue  # Skip this triplet if any peaks are too close

                        # Calculate ratios between the three peaks
                        if radar_detection_mode:
                            ratio_1 = peaks[k] / peaks[j]  # L1 to L3
                            ratio_2 = peaks[j] / (peaks[i] + small_peak_cal)  # L3 to L2

                            # Check if all ratios match the expected ratios within the tolerance
                            if (
                                abs(ratio_1 - expected_ratios[0]) < tolerance and
                                abs(ratio_2 - expected_ratios[1]) < tolerance
                            ):
                                # if largest_peak in (peaks[i], peaks[j], peaks[k]):
                                amplitude_i = peak_amplitudes[np.where(peaks == peaks[i])[0][0]]
                                amplitude_j = peak_amplitudes[np.where(peaks == peaks[j])[0][0]]
                                amplitude_k = peak_amplitudes[np.where(peaks == peaks[k])[0][0]]

                                amplitude_prod = amplitude_i * amplitude_j * amplitude_k

                                if amplitude_prod > max_amplitude_prod:
                                    max_amplitude_prod = amplitude_prod
                                    best_triplet = (peaks[i], peaks[j], peaks[k])
                        else:
                            # If radar is detected, we need to check for the third ratio for a more strict check
                            # Calculate ratios between the three peaks
                            ratio_1 = peaks[k] / peaks[j]  # L1 to L3
                            ratio_2 = peaks[j] / (peaks[i] + small_peak_cal)  # L3 to L2
                            ratio_3 = peaks[k] / (peaks[i] + small_peak_cal)  # L1 to L2

                            # Check if all ratios match the expected ratios within the tolerance
                            if (
                                abs(ratio_1 - expected_ratios[0]) < tolerance and
                                abs(ratio_2 - expected_ratios[1]) < tolerance and
                                abs(ratio_3 - expected_ratios[2]) < tolerance
                            ):
                                amplitude_i = peak_amplitudes[np.where(peaks == peaks[i])[0][0]]
                                amplitude_j = peak_amplitudes[np.where(peaks == peaks[j])[0][0]]
                                amplitude_k = peak_amplitudes[np.where(peaks == peaks[k])[0][0]]

                                if fix_amplitude_order:
                                    if amplitude_k < amplitude_j or amplitude_k < amplitude_i:
                                        continue

                                amplitude_prod = amplitude_i * amplitude_j * amplitude_k

                                # Check if this is the largest amplitude sum found so far
                                if amplitude_prod > max_amplitude_prod:
                                    max_amplitude_prod = amplitude_prod
                                    best_triplet = (peaks[i], peaks[j], peaks[k])
        if best_triplet and not radar_detection_mode:
            valid_peaks = list(best_triplet)
            if len(valid_peaks) != 3:
                predicted_freqs = []
            else:
                valid_peaks = sorted(valid_peaks)
                sub_peak1 = valid_peaks[-1]
                sub_peak2 = valid_peaks[-3]
                sub_peak3 = valid_peaks[-2]
                predicted_freqs = [frequency_axis[sub_peak1], frequency_axis[sub_peak2], frequency_axis[sub_peak3]]
                valid_freq_list.append(predicted_freqs)
        else:
            valid_peaks = []  # No valid triplet found
            predicted_freqs = []
        
    return best_triplet, snr, valid_freq_list


def calculate_slope_and_angle(predicted_peaks, delta_L1, delta_L2, delta_L3, d, c, speed_ratio, frequency_offsets=[0, 1300, 0]):
    """
    Calculate the slope and angle based on predicted peaks and other parameters.

    Parameters:
        predicted_peaks (list of float): A list containing the predicted peak frequencies.
        delta_L1 (float): The difference between first set of delay lines
        delta_L2 (float): The difference between second set of delay lines
        delta_L3 (float): The difference between third set of delay lines
        d (float): The distance between the tag antennas.
        c (float): The speed of light.
        speed_ratio (float): The ratio of the speed of the target to the speed of light.
        frequency_offsets (list of float, optional): A list of frequency offsets to be added to the predicted peaks. 
                                                        Defaults to [0, 1300, 0].

    Returns:
        tuple: A tuple containing:
            - predicted_slope (float): The calculated slope based on the input parameters.
            - predicted_angle (float): The calculated angle in degrees based on the input parameters.
    """
    predicted_peak1, predicted_peak2, predicted_peak3 = predicted_peaks[0] + frequency_offsets[0], predicted_peaks[1] + frequency_offsets[1], predicted_peaks[2] + frequency_offsets[2]
    predicted_slope = (predicted_peak1 + predicted_peak3) / ((delta_L1 + delta_L3) / (speed_ratio * c))
    predicted_angle = np.rad2deg(np.arcsin(((predicted_peak2 * c) / predicted_slope - (delta_L2 / speed_ratio)) / (2*d)))
    return predicted_slope, predicted_angle


def calculate_gt_frequency_peak(data_obj, delta_L1, delta_L2, delta_L3, d, c, speed_ratio):
    """
    Calculate the theoretical ground truth frequency peaks based on input parameters.

    Args:
        data_obj (dict): A dictionary containing the following keys:
            - 'angle' (float): The angle in degrees.
            - 'bw' (float): Bandwidth parameter.
            - 'CD' (float): Coefficient of drag.
            - 'slope' (float): Slope parameter for frequency shift calculation.
        delta_L1 (float): The difference between first set of delay lines
        delta_L2 (float): The difference between second set of delay lines
        delta_L3 (float): The difference between first set of delay lines
        d (float): Distance parameter for frequency shift calculation.
        c (float): Speed of light or another constant for frequency calculation.
        speed_ratio (float): Ratio of the object's speed to a reference speed.

    Returns:
        tuple: A tuple containing three theoretical peak frequencies:
            - theory_peak_1 (float): Theoretical frequency for the first peak.
            - theory_peak_2 (float): Theoretical frequency for the second peak.
            - theory_peak_3 (float): Theoretical frequency for the third peak.
    """
    angle = data_obj['angle']
    theta = np.deg2rad(float(angle))
    bw = data_obj['bw']
    cd = data_obj['CD']
    # Theroetical peak frequencies shift
    theta_shift = (d * np.sin(theta) / c) * data_obj['slope']
    theory_peak_1 = get_output_theory(delta_L1/0.0254, bw, cd, speed_ratio=speed_ratio) + theta_shift
    theory_peak_2 = get_output_theory(delta_L2/0.0254, bw, cd, speed_ratio=speed_ratio) - 2 * theta_shift
    theory_peak_3 = get_output_theory(delta_L3/0.0254, bw, cd, speed_ratio=speed_ratio) - theta_shift
    return theory_peak_1, theory_peak_2, theory_peak_3