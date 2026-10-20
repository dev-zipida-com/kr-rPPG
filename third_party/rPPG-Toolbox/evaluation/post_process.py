"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal using Tarvainen's method with sparse solver.

    Uses sparse Cholesky/LU solve instead of dense matrix inversion,
    reducing complexity from O(n^3) to O(n) for banded systems.
    """
    signal_length = input_signal.shape[0]
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length)
    A = scipy.sparse.eye(signal_length) + (lambda_value ** 2) * D.T.dot(D)
    trend = scipy.sparse.linalg.spsolve(A.tocsc(), input_signal)
    detrended_signal = input_signal - trend
    return detrended_signal

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
    # we recommend low_pass=0.75 and high_pass=2.5 instead of the defaults above.
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) using FFT-based
    cross-correlation, reducing complexity from O(n^2) to O(n log n).
        Args:
            pred_signal(np.array): predicted PPG signal
            gt_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = np.squeeze(pred_signal).copy()
    gt = np.squeeze(gt_signal).copy()
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    # Normalize to zero mean, unit variance for Pearson-like correlation
    pred = (pred - np.mean(pred))
    gt = (gt - np.mean(gt))
    pred_std = np.std(pred)
    gt_std = np.std(gt)
    if pred_std < 1e-8 or gt_std < 1e-8:
        return 0.0
    pred = pred / pred_std
    gt = gt / gt_std
    # FFT-based cross-correlation: O(n log n) instead of O(n^2) loop
    corr = scipy.signal.correlate(pred, gt, mode='full', method='fft')
    corr = corr / min_len
    macc = np.max(np.abs(corr))
    return float(macc)

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.6 Hz
        to 3.3 Hz. 

        Ref for low_pass and high_pass filters:
        R. Cassani, A. Tiwari and T. H. Falk, "Optimal filter characterization for photoplethysmography-based pulse rate and 
        pulse power spectrum estimation," 2020 IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada,
        doi: 10.1109/EMBC44109.2020.9175396.

        Note: to more closely match results in the NeurIPS 2023 toolbox paper, we recommend low_pass=0.75 and high_pass=2.5 
        instead of the defaults above.

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz, equals [45, 150] beats per min
        # bandpass filter between [0.6, 3.3] Hz, equals [36, 198] beats per min
        #
        # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
        # we recommend using 0.75 in place of 0.6 and 2.5 in place of 3.3 in the 
        # below line.
        [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR, macc